"""Builds the LangGraph StateGraph.

    triage --(out_of_scope)-------------------------> safe_response --> finalize --> END
       |----(requires_clarification)---------------> clarification --> finalize --> END
       '----(proceed: answerable/escalation)--> retrieval --> generation --> verification
                                                                                  |--(finalize)--> finalize --> END
                                                                                  |--(retry, <=1x)--> generation
                                                                                  '--(safe_failure)--> safe_failure --> finalize --> END

This satisfies: shared typed state, conditional routing, at least one
retry/fallback path (verification -> generation retry -> safe_failure),
clear separation of deterministic nodes (triage/verification/finalize) from
model-backed nodes (retrieval/generation), and loop protection via both an
explicit attempt counter (config.GRAPH_CONFIG.max_generation_attempts) and
LangGraph's own recursion_limit.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from langgraph.graph import END, StateGraph

from . import nodes
from .config import GRAPH_CONFIG
from .embeddings import get_embedder
from .llm import get_generator
from .retriever import Retriever
from .state import AgentState


def build_graph(data_dir: Path, mock: bool = False):
    embedder = get_embedder(mock=mock)
    retriever = Retriever(data_dir=data_dir, embedder=embedder)
    generator = get_generator(mock=mock)

    workflow = StateGraph(AgentState)

    workflow.add_node("triage", nodes.triage_node)
    workflow.add_node("retrieval", partial(nodes.retrieval_node, retriever=retriever))
    workflow.add_node("generation", partial(nodes.generation_node, generator=generator))
    workflow.add_node("verification", nodes.verification_node)
    workflow.add_node("clarification", nodes.clarification_node)
    workflow.add_node("safe_response", nodes.safe_response_node)
    workflow.add_node("safe_failure", nodes.safe_failure_node)
    workflow.add_node("finalize", nodes.finalize_node)

    workflow.set_entry_point("triage")

    workflow.add_conditional_edges(
        "triage",
        nodes.route_after_triage,
        {
            "out_of_scope": "safe_response",
            "requires_clarification": "clarification",
            "proceed": "retrieval",
        },
    )

    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "verification")

    workflow.add_conditional_edges(
        "verification",
        nodes.route_after_verification,
        {
            "finalize": "finalize",
            "retry": "generation",
            "safe_failure": "safe_failure",
        },
    )

    workflow.add_edge("safe_response", "finalize")
    workflow.add_edge("clarification", "finalize")
    workflow.add_edge("safe_failure", "finalize")
    workflow.add_edge("finalize", END)

    compiled = workflow.compile()
    compiled.__orbitdesk_meta__ = {
        "embedder_name": embedder.name,
        "embedder_load_time_s": getattr(embedder, "load_time_seconds", None),
        "generator_name": generator.name,
        "generator_load_time_s": getattr(generator, "load_time_seconds", None),
        "recursion_limit": GRAPH_CONFIG.recursion_limit,
    }
    return compiled, generator


def run_question(app, question_id: str, question: str) -> AgentState:
    initial_state: AgentState = {
        "question_id": question_id,
        "question": question,
        "generation_attempts": 0,
        "node_log": [],
    }
    result = app.invoke(initial_state, config={"recursion_limit": GRAPH_CONFIG.recursion_limit})
    return result
