"""Pydantic model mirroring data/output_schema.json.

Used both to validate the final response before it leaves the graph and to
give the LLM-integration code a concrete typed target to fill in.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Classification(str, Enum):
    ANSWERABLE = "answerable"
    REQUIRES_CLARIFICATION = "requires_clarification"
    REQUIRES_ESCALATION = "requires_escalation"
    OUT_OF_SCOPE = "out_of_scope"
    SAFE_FAILURE = "safe_failure"


class SourceRef(BaseModel):
    source_id: str = Field(..., description="Knowledge-base document ID or resolved-case ID")
    passage: str = Field(..., min_length=1, description="Relevant excerpt or stable passage identifier")


class AgentResponse(BaseModel):
    classification: Classification
    answer: str = Field(..., min_length=1)
    sources: List[SourceRef] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human: bool
    reason: str = Field(..., min_length=1)
    clarification_question: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)
