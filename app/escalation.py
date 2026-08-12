"""
Lightweight heuristics for flagging a conversation for human handoff.

This runs in two places:
  1. Pre-generation: quick keyword/pattern scan of the incoming user message
     (frustrated tone, explicit request for a human, repeated questions).
  2. Post-generation: the LLM's own self-reported confidence score and
     "escalate" flag from generate_response() in app/llm.py.

The final decision in main.py combines both signals. This module intentionally
uses simple, explainable rules rather than a black-box classifier -- swap in
a fine-tuned sentiment/intent classifier here if you need higher precision.
"""
import re
from typing import List, Tuple

FRUSTRATION_PATTERNS = [
    r"\bthis is (ridiculous|useless|a joke|garbage)\b",
    r"\b(worst|terrible|awful) (support|service|experience)\b",
    r"\bnot (helpful|working|helping)\b",
    r"\bi('?m| am) (so |really |very )?(frustrated|angry|upset|furious|annoyed)\b",
    r"\bwaste of (my )?time\b",
    r"\b(fix this|do something) (now|immediately)\b",
    r"\bunacceptable\b",
    r"!!\s*!*",  # multiple exclamation marks
]

HUMAN_REQUEST_PATTERNS = [
    r"\btalk to (a |an )?(human|person|agent|representative|someone)\b",
    r"\bspeak (to|with) (a |an )?(human|person|agent|representative)\b",
    r"\bcustomer service (rep|representative|agent)\b",
    r"\breal (person|human)\b",
    r"\bescalate\b",
    r"\bsupervisor|manager\b",
]

REPEAT_QUESTION_PATTERNS = [
    r"\bi already (asked|said|told you)\b",
    r"\bfor the (second|third|\d+(st|nd|rd|th)) time\b",
    r"\bagain\b.*\?",
    r"\byou'?re not (listening|understanding)\b",
]

_ALL = [
    ("frustrated_tone", FRUSTRATION_PATTERNS),
    ("human_requested", HUMAN_REQUEST_PATTERNS),
    ("repeated_question", REPEAT_QUESTION_PATTERNS),
]


def detect_frustration(message: str) -> Tuple[bool, List[str]]:
    """Returns (should_escalate, matched_reasons)."""
    lowered = message.lower()
    reasons = []
    for reason, patterns in _ALL:
        for pattern in patterns:
            if re.search(pattern, lowered):
                reasons.append(reason)
                break
    return (len(reasons) > 0, reasons)


def combine_escalation_signals(
    heuristic_flag: bool,
    heuristic_reasons: List[str],
    model_confidence: float,
    model_flagged: bool,
    confidence_threshold: float,
) -> Tuple[bool, str]:
    """
    Combine the pre-generation heuristic with the model's own self-assessment
    into a single decision + human-readable reason string.
    """
    reasons = list(heuristic_reasons)
    if model_flagged:
        reasons.append("model_flagged")
    if model_confidence < confidence_threshold:
        reasons.append("low_confidence")

    escalate = heuristic_flag or model_flagged or model_confidence < confidence_threshold
    reason_str = ", ".join(reasons) if reasons else "n/a"
    return escalate, reason_str
