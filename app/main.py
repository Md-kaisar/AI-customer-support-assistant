"""
FastAPI application entrypoint.

Endpoints:
  POST /chat                  -> core RAG chat pipeline
  POST /voice                 -> audio in -> Whisper transcription -> same chat pipeline -> (optional TTS out)
  GET  /conversations/{id}    -> fetch stored conversation history (debugging/testing aid)
  GET  /escalations           -> list conversations flagged for human handoff
  GET  /health                -> liveness check
  GET  /                      -> serves the static test page

Run with: uvicorn app.main:app --reload

On startup, the knowledge base is auto-seeded if the vector store is empty.
This matters on platforms with ephemeral disks (e.g. Render's free tier),
where CHROMA_PERSIST_DIR is wiped on every deploy/restart -- without this,
the KB would silently be empty after every redeploy until someone remembers
to rerun `python -m app.seed_kb` manually.
"""
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    Message,
    SourceChunk,
    VoiceChatResponse,
    EscalationSummary,
)
from app import db
from app import llm
from app import vector_store
from app import escalation
from app.knowledge_base.faq_docs import FAQ_DOCS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_ai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if vector_store.collection_count() == 0:
            logger.info("Vector store is empty -- auto-seeding knowledge base on startup...")
            n = vector_store.upsert_docs(FAQ_DOCS)
            logger.info("Auto-seeded %d chunks from %d FAQ docs.", n, len(FAQ_DOCS))
        else:
            logger.info("Vector store already has %d chunks; skipping auto-seed.", vector_store.collection_count())
    except Exception:
        # Don't crash the whole app if seeding fails (e.g. transient network
        # issue downloading the local embedding model on first boot) -- log
        # it loudly and let /health and manual `python -m app.seed_kb` surface it.
        logger.exception("Knowledge base auto-seed failed on startup")
    yield


app = FastAPI(
    title="AI Customer Support Assistant",
    description="RAG-based support chatbot with chat + voice channels and escalation logic.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

TTS_OUTPUT_DIR = tempfile.gettempdir()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_chat_pipeline(conversation_id: str, user_message: str) -> ChatResponse:
    """
    The core RAG pipeline, shared by both /chat and /voice:
      1. Heuristic frustration/human-request scan on the raw user message.
      2. Retrieve conversation history from MongoDB (conversation memory).
      3. Retrieve top-k relevant KB chunks from the vector store.
      4. Call the LLM, grounded in history + retrieved context.
      5. Combine heuristic + model self-assessment into an escalation decision.
      6. Persist both turns (user, assistant) to MongoDB.
    """
    heuristic_flag, heuristic_reasons = escalation.detect_frustration(user_message)

    history = await db.get_history(conversation_id, settings.MAX_HISTORY_TURNS)
    retrieved_chunks = vector_store.retrieve(user_message)

    response_text, confidence, model_flagged = llm.generate_response(
        history=history,
        user_message=user_message,
        context_chunks=retrieved_chunks,
    )

    should_escalate, reason = escalation.combine_escalation_signals(
        heuristic_flag=heuristic_flag,
        heuristic_reasons=heuristic_reasons,
        model_confidence=confidence,
        model_flagged=model_flagged,
        confidence_threshold=settings.ESCALATION_CONFIDENCE_THRESHOLD,
    )

    timestamp = _now()
    await db.append_messages(
        conversation_id,
        [
            {"role": "user", "content": user_message, "timestamp": timestamp},
            {"role": "assistant", "content": response_text, "timestamp": _now()},
        ],
    )

    if should_escalate:
        await db.set_escalated(conversation_id, True)
        await db.log_escalation(conversation_id, reason, user_message)
    else:
        await db.set_escalated(conversation_id, False)

    return ChatResponse(
        conversation_id=conversation_id,
        response=response_text,
        confidence=round(confidence, 3),
        escalate=should_escalate,
        escalation_reason=reason if should_escalate else None,
        sources=[SourceChunk(**c) for c in retrieved_chunks],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "vector_store_chunks": vector_store.collection_count()}


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not settings.CHAT_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="No chat API key configured. Set CHAT_API_KEY (or OPENAI_API_KEY) in .env.",
        )
    try:
        return await run_chat_pipeline(req.conversation_id, req.message)
    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {e}")


@app.post("/voice", response_model=VoiceChatResponse)
async def voice(
    audio: UploadFile = File(...),
    conversation_id: str = None,
    respond_with_audio: bool = False,
):
    """
    Accepts an audio file, transcribes it with Whisper, runs the transcript
    through the same RAG chat pipeline used by /chat, and returns the text
    response. Optionally also returns a URL to a synthesized TTS audio reply.
    """
    if not settings.CHAT_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="No chat API key configured. Set CHAT_API_KEY (or OPENAI_API_KEY) in .env.",
        )
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=501,
            detail=(
                "Voice transcription requires a real OpenAI API key set as "
                "OPENAI_API_KEY in .env (OpenRouter keys do not support Whisper). "
                "Text chat via /chat works without this."
            ),
        )

    conversation_id = conversation_id or str(uuid.uuid4())

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = llm.transcribe_audio(tmp_path)
    except Exception as e:
        logger.exception("Whisper transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.remove(tmp_path)

    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the audio.")

    try:
        chat_result = await run_chat_pipeline(conversation_id, transcript)
    except Exception as e:
        logger.exception("Error in /voice chat pipeline")
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {e}")

    audio_url = None
    if respond_with_audio:
        try:
            out_name = f"reply_{uuid.uuid4().hex}.mp3"
            out_path = os.path.join(TTS_OUTPUT_DIR, out_name)
            llm.synthesize_speech(chat_result.response, out_path)
            audio_url = f"/voice/audio/{out_name}"
        except Exception:
            logger.exception("TTS synthesis failed; returning text-only response")

    return VoiceChatResponse(
        **chat_result.model_dump(),
        transcript=transcript,
        audio_response_url=audio_url,
    )


@app.get("/voice/audio/{filename}")
async def get_voice_audio(filename: str):
    path = os.path.join(TTS_OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found or expired.")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=[Message(**m) for m in convo.get("messages", [])],
        escalated=convo.get("escalated", False),
    )


@app.get("/escalations", response_model=list[EscalationSummary])
async def get_escalations():
    docs = await db.list_escalations()
    return [
        EscalationSummary(
            conversation_id=d["conversation_id"],
            reason=d["reason"],
            last_message=d["last_message"],
            flagged_at=d["flagged_at"],
        )
        for d in docs
    ]