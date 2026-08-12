"""
Centralized configuration, loaded from environment variables (.env).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Chat / generation ---
    # Works with OpenAI directly, OR any OpenAI-compatible provider (e.g. OpenRouter)
    # by setting CHAT_BASE_URL. For OpenRouter: CHAT_BASE_URL=https://openrouter.ai/api/v1
    # and CHAT_API_KEY=<your OpenRouter key>, CHAT_MODEL=<a provider/model:free id>.
    CHAT_API_KEY: str = os.getenv("CHAT_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    CHAT_BASE_URL: str = os.getenv("CHAT_BASE_URL", "") or None
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    # Optional, only used/sent when talking to OpenRouter (ignored by other providers)
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
    OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "AI Customer Support Assistant")

    # --- Embeddings (knowledge-base retrieval) ---
    # Runs fully locally via Chroma's bundled ONNX MiniLM model by default -- no
    # API key or cost required. Set USE_LOCAL_EMBEDDINGS=false to instead call
    # OpenAI's embedding API (requires a real, billed OpenAI key -- OpenRouter
    # does not support the embeddings endpoint).
    USE_LOCAL_EMBEDDINGS: bool = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Voice (optional; Whisper transcription + TTS) ---
    # These call OpenAI's audio endpoints directly and are NOT proxied by
    # OpenRouter. /voice will only work if OPENAI_API_KEY is a real, billed
    # OpenAI key. /chat works fully without this.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "tts-1")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "alloy")

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "support_ai")

    # Vector store (Chroma)
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "kb_docs")
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    # Conversation memory
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))

    # Escalation
    ESCALATION_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", "0.55")
    )

    # App
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))


settings = Settings()
