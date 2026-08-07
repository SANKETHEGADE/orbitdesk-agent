"""Deterministic triage classification.

The assignment explicitly asks for "clear separation between deterministic
code and model reasoning". Triage here is intentionally rule-based (regex
over the raw question) rather than another model call: it is cheap, fully
explainable, easy to unit test, and matches the kind of guardrail logic a
real support product would want to audit. Retrieval and generation are
where the Hugging Face models do the actual reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

OUT_OF_SCOPE_PATTERNS = [
    r"\brefund\b",
    r"\bcancel (my|the) subscription\b",
    r"\blegal advice\b",
    r"\bmedical advice\b",
    r"\bfinancial advice\b",
    r"\bsue\b|\blawsuit\b",
    r"\bignore (the )?(supplied|above|previous) (documentation|instructions)\b",
    r"\bdisregard (the )?(documentation|instructions|rules)\b",
]

PROMPT_INJECTION_PATTERNS = [
    r"\bignore (the )?(supplied|above|previous) (documentation|instructions)\b",
    r"\bdisregard (the )?(documentation|instructions|rules)\b",
    r"\byou (are|must) now\b",
    r"\bact as\b.*\b(unrestricted|no rules|dan)\b",
]

ESCALATION_PATTERNS = [
    r"\bescalat\w*\b",
    r"\balready (checked|tried|attempted)\b",
    r"\btwo (export runs?|attempts?|runs?) in a row\b",
    r"\bfailed twice\b",
    r"\btried everything\b",
    r"\bsuggested solution did not work\b",
]

VAGUE_TRIGGER_PATTERNS = [
    r"\bnot working\b",
    r"\bisn'?t working\b",
    r"\bdoesn'?t work\b",
    r"\bstopped working\b",
    r"\bbroken\b",
    r"\bplease fix\b",
]

.
SPECIFIC_SIGNAL_PATTERNS = [
    r"\btimezone\b",
    r"\bapi credential\b",
    r"\bviewer\b|\banalyst\b|\badmin\b|\bowner\b",
    r"\berror code\b|\brender_failed\b|\bsource_refresh_timeout\b|\bdestination_unverified\b|\bowner_access_revoked\b",
    r"\bschedule id\b|\bworkspace id\b|\bconnection id\b|\bdashboard id\b",
    r"\baudit log\b",
]


@dataclass
class TriageResult:
    classification: str
    reason: str
    flags: List[str] = field(default_factory=list)


def _matches_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify(question: str) -> TriageResult:
    flags: List[str] = []

    if _matches_any(PROMPT_INJECTION_PATTERNS, question):
        flags.append("prompt_injection_attempt")

    if _matches_any(OUT_OF_SCOPE_PATTERNS, question):
        return TriageResult(
            classification="out_of_scope",
            reason="Question requests an unsupported action (billing/legal/account change) "
            "or attempts to override the assistant's rules.",
            flags=flags,
        )

    if _matches_any(ESCALATION_PATTERNS, question):
        return TriageResult(
            classification="requires_escalation",
            reason="Question indicates documented troubleshooting was already attempted "
            "and/or explicitly asks about escalation.",
            flags=flags,
        )

    has_vague_complaint = _matches_any(VAGUE_TRIGGER_PATTERNS, question)
    has_specific_signal = _matches_any(SPECIFIC_SIGNAL_PATTERNS, question)
    word_count = len(question.split())
    if has_vague_complaint and not has_specific_signal and word_count < 25:
        flags.append("vague_symptom")
        return TriageResult(
            classification="requires_clarification",
            reason="Question reports a generic symptom without the object, error code, "
            "or ID needed to choose a documented troubleshooting path.",
            flags=flags,
        )

    return TriageResult(
        classification="answerable",
        reason="Question describes a specific, in-scope OrbitDesk behaviour that the "
        "knowledge base is expected to cover.",
        flags=flags,
    )
