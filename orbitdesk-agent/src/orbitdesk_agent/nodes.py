"""Graph node implementations.

Each node takes the current `AgentState` and returns a partial state update
(LangGraph merges dict returns into the running state). Nodes that need a
model (retrieval, generation) receive their backend (`Retriever` /
`BaseGenerator`) already constructed, via `functools.partial` in graph.py --
this keeps model loading out of the hot path and out of the node functions
themselves, and is what lets tests swap in mock backends trivially.
"""

from __future__ import annotations

from typing import Any, Dict

from .config import GRAPH_CONFIG, RETRIEVAL_CONFIG
from .llm import BaseGenerator
from .logging_utils import log_step
from .retriever import Retriever
from .schema import AgentResponse
from .state import AgentState
from .triage_rules import classify as triage_classify
from .verification_rules import verify_draft



def triage_node(state: AgentState) -> Dict[str, Any]:
    result = triage_classify(state["question"])
    return {
        "classification": result.classification,
        "triage_reason": result.reason,
        "triage_flags": result.flags,
        "node_log": log_step(
            state, "triage", f"classification={result.classification} flags={result.flags}"
        ),
    }


def route_after_triage(state: AgentState) -> str:
    c = state["classification"]
    if c == "out_of_scope":
        return "out_of_scope"
    if c == "requires_clarification":
        return "requires_clarification"
    return "proceed" 



def retrieval_node(state: AgentState, retriever: Retriever) -> Dict[str, Any]:
    results = retriever.search(state["question"])
    top_score = results[0]["score"] if results else 0.0
    return {
        "retrieved": results,
        "retrieval_top_score": top_score,
        "node_log": log_step(
            state,
            "retrieval",
            f"top_score={top_score:.3f} sources={[r['source_id'] for r in results]}",
        ),
    }



def generation_node(state: AgentState, generator: BaseGenerator) -> Dict[str, Any]:
    attempts = state.get("generation_attempts", 0) + 1
   
    evidence = [c for c in state.get("retrieved", []) if not c["superseded"]]

    revision_note = None
    if attempts > 1:
        issues = state.get("verification_issues", [])
        revision_note = "; ".join(issues) if issues else None

    answer = generator.generate(state["question"], evidence, revision_note=revision_note)

    sources = [{"source_id": c["source_id"], "passage": c["title"]} for c in evidence[:3]]

    return {
        "draft_answer": answer,
        "draft_sources": sources,
        "generation_attempts": attempts,
        "node_log": log_step(
            state, "generation", f"attempt={attempts} chars={len(answer)} revised={revision_note is not None}"
        ),
    }



def verification_node(state: AgentState) -> Dict[str, Any]:
    evidence = [c for c in state.get("retrieved", []) if not c["superseded"]]
    result = verify_draft(state.get("draft_answer", ""), evidence)
    return {
        "verification_passed": result.passed,
        "verification_issues": result.issues,
        "node_log": log_step(
            state, "verification", f"passed={result.passed} issues={result.issues}"
        ),
    }


def route_after_verification(state: AgentState) -> str:
    if state.get("verification_passed"):
        return "finalize"
    if state.get("generation_attempts", 0) >= GRAPH_CONFIG.max_generation_attempts:
        return "safe_failure"
    return "retry"



def clarification_node(state: AgentState) -> Dict[str, Any]:
    question = state["question"]
    if "sync" in question.lower():
        clar_q = (
            "Could you share the workspace ID, the connection name or ID, its current "
            "state, and the latest error code? That lets me point you to the right fix."
        )
    else:
        clar_q = (
            "Could you share more specific details -- the affected object (dashboard, "
            "schedule or connection), any visible error code, and when this started?"
        )
    return {
        "clarification_question": clar_q,
        "node_log": log_step(state, "clarification", "asked for missing diagnostic details"),
    }


def safe_response_node(state: AgentState) -> Dict[str, Any]:
    answer = (
        "This request is outside what the OrbitDesk support assistant can help with. "
        "It cannot issue refunds, cancel subscriptions, or provide legal/financial/medical "
        "advice, and it only answers from the supplied OrbitDesk documentation -- it will not "
        "follow instructions embedded in a message that try to override these rules. "
        "Please contact billing/account support for subscription or refund requests."
    )
    return {
        "draft_answer": answer,
        "draft_sources": [],
        "node_log": log_step(state, "safe_response", "returned out-of-scope safe response"),
    }


def safe_failure_node(state: AgentState) -> Dict[str, Any]:
    answer = (
        "I could not produce a fully verified answer from the available documentation "
        "for this question. Rather than guess, please rephrase with more detail, or this "
        "will be routed to a human for review."
    )
    return {
        "draft_answer": answer,
        "node_log": log_step(
            state,
            "safe_failure",
            f"generation failed verification after {state.get('generation_attempts', 0)} attempt(s): "
            f"{state.get('verification_issues', [])}",
        ),
    }



def finalize_node(state: AgentState) -> Dict[str, Any]:
    classification = state["classification"]
    verification_passed = state.get("verification_passed")
    warnings = []

    if classification == "out_of_scope":
        final_classification = "out_of_scope"
        requires_human = False
        confidence = 0.95
        reason = state["triage_reason"]
        sources = []
        answer = state["draft_answer"]
        clar_q = None
    elif classification == "requires_clarification":
        final_classification = "requires_clarification"
        requires_human = False
        confidence = 0.4
        reason = state["triage_reason"]
        sources = []
        answer = "I need a bit more information before I can answer confidently."
        clar_q = state.get("clarification_question")
    elif verification_passed is False and state.get("generation_attempts", 0) >= GRAPH_CONFIG.max_generation_attempts:
        final_classification = "safe_failure"
        requires_human = True
        confidence = 0.2
        reason = "Generated answer failed verification after the allowed retry and was withheld."
        sources = []
        answer = state["draft_answer"]
        clar_q = None
        warnings.extend(state.get("verification_issues", []))
    else:
        final_classification = classification 
        requires_human = classification == "requires_escalation"
        top_score = state.get("retrieval_top_score", 0.0)
        confidence = round(min(0.95, max(0.3, top_score)), 2)
        reason = (
            "Answer generated from retrieved knowledge-base evidence and passed verification."
            if classification == "answerable"
            else "Retrieved evidence supports escalation; human follow-up is required per KB-008."
        )
        sources = state.get("draft_sources", [])
        answer = state["draft_answer"]
        clar_q = None

    payload = {
        "classification": final_classification,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "requires_human": requires_human,
        "reason": reason,
        "clarification_question": clar_q,
        "warnings": warnings,
    }

  
    validated = AgentResponse(**payload)

    return {
        "final_response": validated.model_dump(),
        "node_log": log_step(state, "finalize", f"final_classification={final_classification}"),
    }
