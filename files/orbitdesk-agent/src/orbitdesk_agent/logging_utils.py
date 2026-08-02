"""Small helper so every node appends a structured, human-readable log line
to state['node_log'] as it runs. This is what the video walkthrough and the
CLI print to satisfy "logs showing which nodes executed".
"""

from __future__ import annotations

from typing import Any, Dict, List


def log_step(state: Dict[str, Any], node: str, detail: str) -> List[Dict[str, str]]:
    entries = list(state.get("node_log", []))
    entries.append({"node": node, "detail": detail})
    return entries
