"""Deterministic verification of a generated draft answer.

Again intentionally rule-based rather than an extra LLM call: verification
needs to be trustworthy and explainable, so it checks concrete, checkable
things (citations present and real, no claimed actions the system cannot
perform, evidence non-empty) rather than asking a model to grade itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .state import RetrievedChunk

CITATION_RE = re.compile(r"\[([A-Z]+-\d+)\]")

UNSUPPORTED_ACTION_PHRASES = [
    r"\bi(?:'ve| have) (issued|processed) (the |your )?refund\b",
    r"\bi(?:'ve| have) (created|generated|issued) (the |your |a )?(api )?credential\b",
    r"\bi(?:'ve| have) changed (your|the) (workspace )?setting\b",
    r"\bi(?:'ve| have) (cancelled|canceled) (your|the) subscription\b",
    r"\bi(?:'ve| have) contacted\b",
    r"\byour refund (has been|is) (issued|processed)\b",
]


@dataclass
class VerificationResult:
    passed: bool
    issues: List[str] = field(default_factory=list)


def verify_draft(answer: str, evidence: List[RetrievedChunk]) -> VerificationResult:
    issues: List[str] = []

    if not answer or not answer.strip():
        issues.append("empty_answer")

    cited_ids = set(CITATION_RE.findall(answer))
    retrieved_ids = {c["source_id"] for c in evidence}

    if not cited_ids:
        issues.append("no_source_citations")
    else:
        hallucinated = cited_ids - retrieved_ids
        if hallucinated:
            issues.append(f"citation_not_in_retrieved_evidence:{','.join(sorted(hallucinated))}")

    superseded_cited = {c["source_id"] for c in evidence if c["superseded"]} & cited_ids
    if superseded_cited:
        issues.append(f"cites_superseded_case_as_current:{','.join(sorted(superseded_cited))}")

    for pattern in UNSUPPORTED_ACTION_PHRASES:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            issues.append("claims_unsupported_action")
            break

    if not evidence:
        issues.append("no_retrieved_evidence")

    return VerificationResult(passed=len(issues) == 0, issues=issues)
