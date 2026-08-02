# OrbitDesk Support Agent Network

A local-first, graph-orchestrated support agent for the fictional OrbitDesk
product, built for the AI Engineer Internship assignment. Triage, retrieval,
generation and verification are implemented as LangGraph nodes over shared
typed state, with a bounded retry loop and safe-failure fallback.

## AI-assistant disclosure

I used an AI coding assistant (Claude) while building this project. It helped
scaffold the LangGraph node/graph structure, the pydantic schema, the
regex-based triage/verification rule sets, the pytest routing tests, and this
README. I designed the overall architecture (triage → retrieval → generation
→ verification → finalize, with the retry/safe-failure branches), wrote and
reviewed the prompts and rules, ran and read every test, and can explain or
modify any part of this implementation.

*(Replace this paragraph with your own accurate account of what you did
yourself vs. what the assistant helped with before you submit — that's the
whole point of the disclosure requirement.)*

## Architecture

![Graph diagram](graph_diagram.png)

```
question
   │
   ▼
┌─────────┐  out_of_scope   ┌───────────────┐
│ Triage  ├────────────────▶│ Safe response │──┐
└────┬────┘                 └───────────────┘  │
     │ requires_clarification                  │
     │             ┌────────────────┐          │
     ├────────────▶│ Clarification  │──────────┤
     │             └────────────────┘          │
     │ answerable / requires_escalation         │
     ▼                                          │
┌───────────┐   ┌────────────┐   ┌──────────────┐
│ Retrieval ├──▶│ Generation ├──▶│ Verification │
└───────────┘   └─────▲──────┘   └──────┬───────┘
                       │  retry (<=1x)   │ passed
                       └─────────────────┤
                                          │ failed, attempts exhausted
                                          ▼
                                   ┌──────────────┐
                                   │ Safe failure │──┐
                                   └──────────────┘  │
                                                      ▼
                                              ┌──────────────┐
                                              │   Finalize   │──▶ structured JSON
                                              └──────────────┘
```

| Node | Type | Responsibility |
|---|---|---|
| `triage` | deterministic | Classifies the request via regex rules (out-of-scope / prompt-injection, escalation cues, vague-symptom cues, else answerable). |
| `retrieval` | model-backed | Embeds the question and all KB/case chunks with a local sentence-transformers model; cosine-similarity top-k search, no vector DB. |
| `generation` | model-backed | Local Hugging Face causal LM answers strictly from the retrieved (non-superseded) evidence, citing `[source_id]`. |
| `verification` | deterministic | Checks citations exist, citations refer to actually-retrieved evidence, no superseded case is presented as current, no claimed unsupported action, non-empty evidence/answer. |
| `clarification` / `safe_response` / `safe_failure` | deterministic | Terminal branches for the three routes that don't need (or shouldn't get) a generated answer. |
| `finalize` | deterministic | Assembles and pydantic-validates the schema-conformant JSON response. |

**Loop protection:** `generation_attempts` is tracked in state and capped at
`GRAPH_CONFIG.max_generation_attempts` (default 2) — after one failed
verification the graph revises once; if it fails again it routes to
`safe_failure` rather than looping. LangGraph's own `recursion_limit` is also
set as a second, independent guard (`config.py`).

## Local models used

| Role | Model | Revision | Library |
|---|---|---|---|
| Embedding / retrieval | `sentence-transformers/all-MiniLM-L6-v2` | pinned in `config.py` | `sentence-transformers` |
| Response generation | `Qwen/Qwen2.5-0.5B-Instruct` | `main` | `transformers` |

Both are small enough to run on CPU. Exact model names/revisions live in
`src/orbitdesk_agent/config.py` (`MODEL_CONFIG`) so they're a single source of
truth. **Fill in the table below after your own run:**

| Metric | Value |
|---|---|
| Hardware | Intel Core i5-12450H (12th gen), 16 GB RAM, NVIDIA RTX 3050 4GB (present but **not used** -- see note below) |
| Embedding model load time | ~6.4 s |
| Generation model load time | ~4.8 s (after models are already cached locally; first-ever download run took much longer) |
| Typical end-to-end latency per question | ~0 s for out-of-scope/clarification (no model call); ~9-25 s for a single-pass answerable question; up to ~90 s when verification fails once and generation retries |

**Note on GPU:** a plain `pip install torch` on Windows installs the CPU-only build by default. This run used CPU inference throughout, which explains the latency above. Installing the CUDA-enabled build (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) would let `RealGenerator` use the RTX 3050 automatically (the code already checks `torch.cuda.is_available()`), and would substantially cut generation latency. Left as CPU-only for this submission since it already meets the "CPU-compatible" requirement and keeps setup simpler for reviewers without a GPU.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads the two models from the Hugging Face Hub (needs
network). After that, models are cached locally
(`~/.cache/huggingface`) and the app runs fully offline — turn off networking
and re-run `python cli.py --sample` to confirm.

## Running

```bash
# Real local models (requires the one-time download above)
python cli.py --sample --out sample_outputs/real_run.json

# Ask one question
python cli.py --question "Can a read-only user create API credentials?"

# Mock mode: deterministic, no downloads, no network -- useful for a fast
# sanity check of the graph's routing logic, and is what the automated
# tests use.
python cli.py --sample --mock --out sample_outputs/mock_run.json
```

Every run prints the node-execution log and the final schema-conformant JSON
for each question. `sample_outputs/mock_run.json` in this repo is a captured
mock-mode run showing all five required routes; regenerate `real_run.json`
with your own local models before submitting.

## Tests

```bash
pytest -q
```

`tests/test_graph_routing.py` runs the graph with the mock backend and
asserts on `classification`, the sequence of node names in `node_log`,
`verification_passed`, `generation_attempts`, and the final schema shape --
never on the literal text a model produced. It covers all five required test
cases plus an escalation-routing case and a loop-protection case:

1. `test_directly_answerable_routes_through_full_pipeline` — directly answerable
2. `test_answer_pulls_evidence_from_multiple_documents` — needs two+ documents
3. `test_vague_question_routes_to_clarification` — ambiguous → clarification
4. `test_out_of_scope_request_is_safely_declined` — out-of-scope / prompt injection
5. `test_failed_verification_triggers_retry_then_succeeds` — verification failure → retry → pass
6. `test_escalation_signal_routes_to_escalation_with_requires_human` — escalation path
7. `test_generation_attempts_never_exceed_configured_max` — loop protection
8. `test_final_response_matches_schema_shape[...]` (×4) — schema validation across routes

## Design notes, trade-offs, limitations

- **Triage and verification are rule-based, not model calls.** This was a
  deliberate choice to keep those two safety-relevant steps auditable and
  fast, and to make the routing tests deterministic without mocking a model.
  The trade-off is that triage rules are keyword/regex-based and won't
  generalize to phrasings outside the patterns in `triage_rules.py` — a
  learned classifier (e.g. a small local zero-shot or fine-tuned
  intent-classification model) would generalize better.
- **Retrieval is section-level chunking + cosine similarity**, no reranker.
  With more time, adding a local cross-encoder reranking pass over the top-k
  results would likely improve precision on questions whose evidence spans
  multiple documents (like Q-001).
- **Observed in my own run:** the 0.5B generation model doesn't reliably follow the `[SOURCE-ID]` citation instruction -- on Q-001 it produced two drafts with no bracketed citations at all, and the graph correctly routed to `safe_failure` after the retry also failed rather than returning an unverifiable answer. Q-004 shows the opposite outcome on the same setup: the first draft failed verification, the retry succeeded with proper citations. This is the verification/retry/safe-failure design working as intended, but it also means answer quality is noticeably dependent on model size -- a larger instruction-tuned model (or a stricter structured-output constraint at generation time, e.g. forcing the model to fill a `sources` field directly instead of inline brackets) would likely reduce how often the safe-failure path triggers.
- **Known limitation:** the clarification question is currently templated (two hand-written variants keyed on whether "sync" appears in the question), not generated from retrieved context. With more time I'd have the LLM draft the clarification question, constrained to only ask about fields listed in `KB-008`'s "Information to Collect" section.
- **Known limitation:** verification's citation check only knows how to
  parse `[SOURCE-ID]`-style citations; a generation model that cites
  differently (spells out "according to KB-004" without brackets) would
  incorrectly fail verification. A more robust check would use fuzzy
  matching against retrieved chunk text instead of a fixed citation format.

## Repository layout

```
cli.py                          entry point
requirements.txt
data/                            copy of the supplied assignment material
src/orbitdesk_agent/
  config.py                      model names/revisions, thresholds
  schema.py                      pydantic AgentResponse (mirrors output_schema.json)
  state.py                       shared typed graph state
  kb.py                          KB markdown + resolved_cases.json loader/chunker
  embeddings.py                  real (sentence-transformers) + mock embedder
  llm.py                         real (transformers) + mock generator
  retriever.py                   in-memory cosine-similarity vector index
  triage_rules.py                deterministic triage classifier
  verification_rules.py          deterministic verification checks
  nodes.py                       LangGraph node functions
  graph.py                       StateGraph assembly + conditional routing
tests/test_graph_routing.py      routing tests (mock backend)
scripts/make_graph_diagram.py    regenerates graph_diagram.png
graph_diagram.png
sample_outputs/mock_run.json     captured mock-mode run (all 5 routes)
```
