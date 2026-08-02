#!/usr/bin/env python3
"""Renders the LangGraph workflow as a PNG for submission.

    python scripts/make_graph_diagram.py
"""
from graphviz import Digraph

g = Digraph("orbitdesk_agent_graph", format="png")
g.attr(rankdir="TB", fontname="Helvetica", bgcolor="white", splines="true", nodesep="0.4", ranksep="0.55")
g.attr("node", fontname="Helvetica", fontsize="11", shape="box", style="rounded,filled")

deterministic = {"fillcolor": "#E6F1FB", "color": "#185FA5"}
model_backed = {"fillcolor": "#EAF3DE", "color": "#3B6D11"}
terminal = {"fillcolor": "#F1EFE8", "color": "#5F5E5A"}

g.node("start", "question in", shape="ellipse", **terminal)
g.node("triage", "1. Triage\n(deterministic rules)", **deterministic)
g.node("retrieval", "2. Retrieval\n(HF embedding model:\nall-MiniLM-L6-v2)", **model_backed)
g.node("generation", "3. Response generation\n(HF local LLM:\nQwen2.5-0.5B-Instruct)", **model_backed)
g.node("verification", "4. Verification\n(deterministic checks)", **deterministic)
g.node("clarification", "Clarification\nresponse", **terminal)
g.node("safe_response", "Safe out-of-scope\nresponse", **terminal)
g.node("safe_failure", "Safe failure\nresponse", **terminal)
g.node("finalize", "Finalize\n(schema validation)", **deterministic)
g.node("end", "structured JSON out", shape="ellipse", **terminal)

g.edge("start", "triage")
g.edge("triage", "safe_response", label="  out_of_scope")
g.edge("triage", "clarification", label="  requires_clarification")
g.edge("triage", "retrieval", label="  answerable /\n  requires_escalation")

g.edge("retrieval", "generation")
g.edge("generation", "verification")
g.edge("verification", "finalize", label="  passed")
g.edge("verification", "generation", label="  failed, attempts < 2\n  (retry, revise draft)")
g.edge("verification", "safe_failure", label="  failed, attempts >= 2\n  (max retries reached)")

g.edge("clarification", "finalize")
g.edge("safe_response", "finalize")
g.edge("safe_failure", "finalize")
g.edge("finalize", "end")

out_path = g.render("graph_diagram", directory="/home/claude/orbitdesk-agent", cleanup=True)
print("wrote", out_path)
