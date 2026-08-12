"""
Pydantic schemas for request/response bodies.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Client-generated conversation id (UUID recommended)")
    message: str = Field(..., min_length=1, description="The user's message")
    customer_id: Optional[str] = Field(None, description="Optional customer/account identifier")


class SourceChunk(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    confidence: float
    escalate: bool
    escalation_reason: Optional[str] = None
    sources: List[SourceChunk] = []


class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Message]
    escalated: bool = False


class VoiceChatResponse(ChatResponse):
    transcript: str
    audio_response_url: Optional[str] = None


class EscalationSummary(BaseModel):
    conversation_id: str
    reason: str
    last_message: str
    flagged_at: str
