#!/usr/bin/env python3
"""CLI for the OrbitDesk support agent network.

Examples
--------
Run all five sample questions with the mock backend (no downloads, no
network -- good for a quick sanity check or for grading environments
without internet access to the Hugging Face Hub):

    python cli.py --sample --mock

Run all five sample questions with the real local Hugging Face models
(requires the models to have been downloaded once; can then run fully
offline):

    python cli.py --sample

Ask a single ad-hoc question:

    python cli.py --question "Can a read-only user create API credentials?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orbitdesk_agent.graph import build_graph, run_question  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def print_log(state: dict) -> None:
    print("\n--- node execution log ---")
    for entry in state.get("node_log", []):
        print(f"  [{entry['node']}] {entry['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OrbitDesk local support agent")
    parser.add_argument("--question", type=str, help="Ask a single ad-hoc question")
    parser.add_argument("--sample", action="store_true", help="Run all sample_questions.json")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock embedder/generator instead of downloading real HF models",
    )
    parser.add_argument("--out", type=str, help="Write JSON results to this file")
    args = parser.parse_args()

    print(f"Building graph (mock={args.mock}) ...")
    t0 = time.time()
    app, generator = build_graph(DATA_DIR, mock=args.mock)
    print(f"Graph + models ready in {time.time() - t0:.2f}s")
    print("Model info:", json.dumps(app.__orbitdesk_meta__, indent=2))

    outputs = []

    def run_and_report(qid: str, question: str) -> None:
        print(f"\n=== {qid}: {question}")
        t0 = time.time()
        state = run_question(app, qid, question)
        latency = time.time() - t0
        print_log(state)
        print(f"\n--- final structured response (latency={latency:.2f}s) ---")
        print(json.dumps(state["final_response"], indent=2))
        outputs.append(
            {
                "question_id": qid,
                "question": question,
                "latency_seconds": round(latency, 3),
                "response": state["final_response"],
                "node_log": state.get("node_log", []),
            }
        )

    if args.question:
        run_and_report("AD-HOC", args.question)
    elif args.sample:
        sample = json.loads((DATA_DIR / "sample_questions.json").read_text())
        for item in sample["questions"]:
            run_and_report(item["question_id"], item["question"])
    else:
        parser.print_help()
        return

    if args.out:
        Path(args.out).write_text(json.dumps(outputs, indent=2))
        print(f"\nWrote {len(outputs)} result(s) to {args.out}")


if __name__ == "__main__":
    main()
