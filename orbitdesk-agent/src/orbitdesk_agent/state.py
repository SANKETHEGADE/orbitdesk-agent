"""Shared typed state passed between graph nodes.

LangGraph threads a single mutable state object through every node. Keeping
it as an explicit TypedDict (rather than free-form dicts) is what the
assignment means by "shared typed state", and it also gives us one place to
see the full contract between nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RetrievedChunk(TypedDict):
    source_id: str         
    title: str
    text: str
    score: float
    superseded: bool


class NodeLogEntry(TypedDict):
    node: str
    detail: str


class AgentState(TypedDict, total=False):
    
    question_id: str
    question: str

    
    classification: str           
    triage_reason: str
    triage_flags: List[str]       
   
    retrieved: List[RetrievedChunk]
    retrieval_top_score: float

   
    draft_answer: str
    draft_sources: List[Dict[str, str]]
    draft_confidence: float
    generation_attempts: int

    
    verification_passed: bool
    verification_issues: List[str]

    
    final_response: Dict[str, Any]

    
    node_log: List[NodeLogEntry]
    route: str                    
