"""Loads the supplied knowledge-base markdown files and resolved_cases.json,
and splits them into retrievable chunks.

Deterministic, no model calls here -- this is the "deterministic code"
half of the retrieval responsibility. The embedding model only scores the
chunks this module produces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Chunk:
    source_id: str      # KB-001, CASE-1041, ...
    title: str           # document title or case title
    text: str            # chunk body used for embedding + shown as evidence
    superseded: bool = False


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_block, body = m.group(1), m.group(2)
    meta = {}
    for line in fm_block.splitlines():
        if ":" in line and not line.strip().startswith("["):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body


def load_kb_chunks(kb_dir: Path) -> List[Chunk]:
    """Split each KB markdown file into one chunk per '##' section.

    Section-level chunks keep evidence focused (e.g. only the
    "Troubleshooting" section of KB-004) rather than dumping whole documents
    into the prompt.
    """
    chunks: List[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        doc_id = meta.get("document_id", path.stem)
        doc_title = meta.get("title", path.stem)

        # Split on level-2 headings; keep the doc-level intro (before first ##)
        # as its own chunk too.
        parts = re.split(r"\n(?=## )", body.strip())
        for part in parts:
            part = part.strip()
            if not part:
                continue
            heading_match = re.match(r"^##\s+(.*)", part)
            section_title = heading_match.group(1).strip() if heading_match else doc_title
            chunks.append(
                Chunk(
                    source_id=doc_id,
                    title=f"{doc_title} — {section_title}" if heading_match else doc_title,
                    text=part,
                    superseded=False,
                )
            )
    return chunks


def load_case_chunks(resolved_cases_path: Path) -> List[Chunk]:
    """Turn each resolved case into one retrievable chunk."""
    data = json.loads(resolved_cases_path.read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    for case in data.get("cases", []):
        superseded = case.get("status") == "superseded"
        lines = [f"Title: {case['title']}", f"Status: {case['status']}"]
        if case.get("symptoms"):
            lines.append("Symptoms: " + "; ".join(case["symptoms"]))
        if case.get("resolution"):
            lines.append("Resolution steps: " + "; ".join(case["resolution"]))
        if case.get("important_limit"):
            lines.append("Important limit: " + case["important_limit"])
        if case.get("superseded_reason"):
            lines.append("Superseded reason: " + case["superseded_reason"])
        text = "\n".join(lines)
        chunks.append(
            Chunk(
                source_id=case["case_id"],
                title=case["title"],
                text=text,
                superseded=superseded,
            )
        )
    return chunks


def load_all_chunks(data_dir: Path) -> List[Chunk]:
    kb_chunks = load_kb_chunks(data_dir / "knowledge_base")
    case_chunks = load_case_chunks(data_dir / "resolved_cases.json")
    return kb_chunks + case_chunks
