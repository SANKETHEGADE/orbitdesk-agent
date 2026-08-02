"""Shared typed state passed between graph nodes.

LangGraph threads a single mutable state object through every node. Keeping
it as an explicit TypedDict (rather than free-form dicts) is what the
assignment means by "shared typed state", and it also gives us one place to
see the full contract between nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RetrievedChunk(TypedDict):
    source_id: str          # KB-00x or CASE-xxxx
    title: str
    text: str
    score: float
    superseded: bool


class NodeLogEntry(TypedDict):
    node: str
    detail: str


class AgentState(TypedDict, total=False):
    # ---- input ----
    question_id: str
    question: str

    # ---- triage output ----
    classification: str            # answerable | requires_clarification | requires_escalation | out_of_scope
    triage_reason: str
    triage_flags: List[str]        # e.g. ["prompt_injection_attempt", "vague_symptom"]

    # ---- retrieval output ----
    retrieved: List[RetrievedChunk]
    retrieval_top_score: float

    # ---- generation output ----
    draft_answer: str
    draft_sources: List[Dict[str, str]]
    draft_confidence: float
    generation_attempts: int

    # ---- verification output ----
    verification_passed: bool
    verification_issues: List[str]

    # ---- final structured output ----
    final_response: Dict[str, Any]

    # ---- orchestration bookkeeping ----
    node_log: List[NodeLogEntry]
    route: str                     # last routing decision, used by conditional edges
