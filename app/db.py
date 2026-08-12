"""
MongoDB connection (async via motor) + conversation memory helpers.

Collections:
  - conversations: one document per conversation_id, holding the full
    message history and escalation state. This is the "memory" layer
    that lets context persist across turns of a chat.
  - escalations: append-only log of conversations flagged for human handoff.
"""
from datetime import datetime, timezone
from typing import List, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings

_client = AsyncIOMotorClient(settings.MONGODB_URI)
_db = _client[settings.MONGODB_DB]

conversations_col = _db["conversations"]
escalations_col = _db["escalations"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_conversation(conversation_id: str) -> Optional[dict]:
    return await conversations_col.find_one({"_id": conversation_id})


async def get_history(conversation_id: str, limit_turns: int) -> List[Dict]:
    """Return the last `limit_turns` (user+assistant pairs) of messages, oldest first."""
    convo = await get_conversation(conversation_id)
    if not convo:
        return []
    messages = convo.get("messages", [])
    # each "turn" is roughly 2 messages (user, assistant)
    max_messages = limit_turns * 2
    return messages[-max_messages:]


async def append_messages(conversation_id: str, new_messages: List[Dict]) -> None:
    """Append one or more messages to a conversation, creating it if needed."""
    await conversations_col.update_one(
        {"_id": conversation_id},
        {
            "$push": {"messages": {"$each": new_messages}},
            "$set": {"updated_at": _now()},
            "$setOnInsert": {"created_at": _now(), "escalated": False},
        },
        upsert=True,
    )


async def set_escalated(conversation_id: str, escalated: bool) -> None:
    await conversations_col.update_one(
        {"_id": conversation_id},
        {"$set": {"escalated": escalated, "updated_at": _now()}},
        upsert=True,
    )


async def log_escalation(conversation_id: str, reason: str, last_message: str) -> None:
    await escalations_col.insert_one(
        {
            "conversation_id": conversation_id,
            "reason": reason,
            "last_message": last_message,
            "flagged_at": _now(),
        }
    )


async def list_escalations(limit: int = 50) -> List[Dict]:
    cursor = escalations_col.find().sort("flagged_at", -1).limit(limit)
    return [doc async for doc in cursor]
