"""
Thin wrapper around the OpenAI API: chat completion (with structured
confidence output), embeddings for the vector store, Whisper transcription,
and optional TTS for voice replies.
"""
import json
import logging
from typing import List, Dict, Tuple

from openai import OpenAI

from app.config import settings

logger = logging.getLogger("support_ai.llm")

# Chat client: talks to OpenAI by default, or any OpenAI-compatible provider
# (e.g. OpenRouter) when CHAT_BASE_URL is set.
_extra_headers = {}
if settings.CHAT_BASE_URL and "openrouter.ai" in settings.CHAT_BASE_URL:
    # Optional but recommended by OpenRouter for attribution/rate-limit purposes.
    _extra_headers = {
        "HTTP-Referer": settings.OPENROUTER_SITE_URL,
        "X-Title": settings.OPENROUTER_SITE_NAME,
    }

client = OpenAI(
    api_key=settings.CHAT_API_KEY,
    base_url=settings.CHAT_BASE_URL,  # None -> OpenAI SDK default (api.openai.com)
    default_headers=_extra_headers or None,
)

# Separate client strictly for OpenAI's audio endpoints (Whisper/TTS), which
# OpenRouter does not proxy. Only used by the optional /voice endpoint.
voice_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None

SYSTEM_PROMPT = """You are a customer support assistant for Acme Cloud Storage.
Answer the user's question using ONLY the provided knowledge base context and
the conversation history. Be concise, friendly, and precise.

Rules:
- If the context does not contain enough information to answer confidently,
  say so honestly rather than guessing, and lower your confidence score.
- Never invent policies, prices, or technical details that are not in the context.
- If the user is angry, confused after repeated attempts, or explicitly asks
  for a human, acknowledge it and recommend escalation.

You MUST respond with a single JSON object with exactly these fields:
{
  "response": "<the natural-language reply to show the user>",
  "confidence": <float between 0.0 and 1.0, how confident you are the reply
                  correctly and fully resolves the user's question>,
  "escalate": <true/false, whether this should be handed off to a human agent>
}
Do not include any text outside the JSON object.
"""


def build_context_block(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant knowledge base articles were found."
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['title']}]\n{c['text']}")
    return "\n\n".join(parts)


def generate_response(
    history: List[Dict],
    user_message: str,
    context_chunks: List[Dict],
) -> Tuple[str, float, bool]:
    """
    Calls the chat completion endpoint with retrieved context + history,
    returns (response_text, confidence, model_flagged_escalate).
    """
    context_block = build_context_block(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(
        {
            "role": "system",
            "content": f"KNOWLEDGE BASE CONTEXT:\n{context_block}",
        }
    )
    # prior turns (already role/content dicts stored in Mongo)
    for m in history:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        # Some OpenAI-compatible providers (many free OpenRouter models included)
        # reject or ignore response_format. Retry once without it -- the prompt
        # already instructs JSON output, and the parser below falls back safely
        # if the model still doesn't comply.
        logger.info("response_format not supported by provider/model (%s); retrying without it", e)
        completion = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=messages,
            temperature=0.3,
        )

    raw = completion.choices[0].message.content or ""
    # Some free/open-weight models wrap JSON in markdown code fences despite
    # instructions not to -- strip those before parsing.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        response_text = parsed.get("response", "").strip()
        confidence = float(parsed.get("confidence", 0.5))
        escalate = bool(parsed.get("escalate", False))
        if not response_text:
            raise ValueError("empty response field")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Failed to parse structured LLM output (%s); falling back", e)
        response_text = raw or "I'm sorry, I wasn't able to generate a reply. Let me connect you with a human agent."
        confidence = 0.3
        escalate = True

    return response_text, confidence, escalate


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Only used if USE_LOCAL_EMBEDDINGS=false in .env. By default, the vector
    store embeds locally via Chroma's bundled model instead (see
    app/vector_store.py) and never calls this function -- so this path
    requires a real, billed OpenAI key (not an OpenRouter key, which does
    not support the embeddings endpoint).
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OpenAI-based embeddings requested (USE_LOCAL_EMBEDDINGS=false) but "
            "OPENAI_API_KEY is not set. Either set USE_LOCAL_EMBEDDINGS=true "
            "(default, free, local) or provide a real OpenAI key."
        )
    embed_client = voice_client or OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = embed_client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def transcribe_audio(file_path: str) -> str:
    """Whisper transcription for the optional /voice endpoint. Requires a
    real OpenAI key -- OpenRouter does not proxy audio endpoints."""
    if voice_client is None:
        raise RuntimeError(
            "Voice transcription requires a real OpenAI API key set as "
            "OPENAI_API_KEY in .env (OpenRouter keys do not support Whisper). "
            "Text chat via /chat does not need this."
        )
    with open(file_path, "rb") as f:
        transcript = voice_client.audio.transcriptions.create(
            model=settings.WHISPER_MODEL,
            file=f,
        )
    return transcript.text


def synthesize_speech(text: str, output_path: str) -> str:
    """Generate TTS audio for a reply. Requires a real OpenAI key -- same
    caveat as transcribe_audio above."""
    if voice_client is None:
        raise RuntimeError(
            "TTS requires a real OpenAI API key set as OPENAI_API_KEY in .env "
            "(OpenRouter keys do not support TTS)."
        )
    with voice_client.audio.speech.with_streaming_response.create(
        model=settings.TTS_MODEL,
        voice=settings.TTS_VOICE,
        input=text,
    ) as response:
        response.stream_to_file(output_path)
    return output_path
