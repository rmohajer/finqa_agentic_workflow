# AGENTS.md - FinQA Agentic Workflow

## Quick Start
```bash
export NVIDIA_API_KEY="your_key"  # Required for LLM (Nemotron 3 Ultra)
export NVIDIA_EMBEDDING_API_KEY="your_key"  # Required for embeddings (Nemotron 3.5 Embed 1B)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp-key.json"  # For Document AI (optional)
uv sync  # or pip install -e ".[test]"
python app.py  # Launches Gradio on localhost:7860 (share=False)
```

## Architecture Essentials
- **Entry point**: `app.py` → Gradio UI + async handlers + session management
- **Agent**: `agentic_system.py` → LangGraph ReAct agent (`finqa_graph`) + `FinQAResponseAgent` wrapper
- **State**: `FinState` (extends `MessagesState`) tracks report_state, user_query[], call_mode, context_content, context_examples, query_cont
- **Two retrieval systems**: User PDF chunks (`df_docs.content_embed`) + ConvFinQA examples (`df_finqa.summary_emb`)
- **Structured output**: Enforced via `model_call_tool` (Pydantic schema: Thought[], References[], Final_answer)
- **Two PDF parsers**: Docling (local, CPU) + Google Document AI (cloud, OCR/Layout/Form)
- **Configuration**: Centralized in `config/config-common-yaml`
- **ConvFinQA embeddings**: Cached in `data/convfinqa_embeddings_cache.npy`, lazy init, pre-computed at startup

## Key Commands
```bash
ruff format .      # Format
ruff check .       # Lint
mypy .             # Type check
pytest tests/parsing_helper_tests.py -v  # Run unit tests
python tests.py parsing  # Run test runner
```

## Critical Gotchas
1. **Global `df_docs` mutation** in `FinQAResponseAgent.predict()` (line 290) — not thread-safe for multi-user
2. **Cache cleaning on startup** (app.py:18-27) — aggressive, removes HF/pip caches
3. **No persistence** — `InMemorySaver` loses state on restart
4. **Single global agent** — `FinQAAgent` shared across users (isolated by `thread_id`)
5. **Model config** — `NVIDIA_MODEL` in `config/config-common-yaml` is single source of truth

## File Responsibilities
| File | Purpose |
|------|---------|
| `app.py` | Gradio UI, PDF upload, chat handlers, parser dropdown, session management |
| `agentic_system.py` | LangGraph nodes, FinState, FinQAResponseAgent, lazy embedding init |
| `prompt_helpers.py` | System/user prompt templates |
| `tool_helpers.py` | Math tools + `model_call_tool` output schema + config loader |
| `parsing_helpers.py` | Docling PDF→MD, Google Document AI (OCR/Layout/Form), MarkdownHeaderTextSplitter, embeddings, similarity search |
| `config/config-common-yaml` | Centralized config (models, chunk length, Document AI) |
| `data/convfinqa_filtered.csv` | Few-shot examples (loaded at startup) |
| `tests/parsing_helper_tests.py` | Unit tests for parsing_helpers.py (81 tests) |
| `tests.py` | Main test runner CLI |
| `scratch/data_parsing_experiment.ipynb` | Data prep pipeline (summaries, embeddings, restructuring) |

## Modification Entry Points
- Add tools → `tool_helpers.py`
- Change prompts → `prompt_helpers.py`
- Modify graph → `agentic_system.py` (StateGraph builder)
- Adjust parsing → `parsing_helpers.py`
- UI changes → `app.py`
- Add tests → `tests/parsing_helper_tests.py`
- Config changes → `config/config-common-yaml`

## Agent Permissions
- **MUST ask before any `edit`, `write`, or `patch` tool call**
- Present: file path, exact change (oldString → newString), reason
- Wait for explicit user confirmation ("yes", "proceed", "go ahead")
- No silent edits — even for "obvious" fixes

## Behavioral Guidelines
**Reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code.**

### 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Environment
- `NVIDIA_API_KEY` — Required for LLM (Nemotron 3 Ultra via `ChatNVIDIA`)
- `NVIDIA_EMBEDDING_API_KEY` — Required for embeddings (Nemotron 3.5 Embed 1B via `NVIDIAEmbeddings`)
- `GROQ_API_KEY` — Fallback LLM (Groq `ChatGroq`)
- `GOOGLE_CLOUD_PROJECT` — GCP project ID for Document AI
- `DOC_AI_PROCESSOR_ID` — Document AI processor ID
- `DOC_AI_LOCATION` — Processor location (e.g., `australia-southeast1`)
- `GOOGLE_APPLICATION_CREDENTIALS` — Path to service account key JSON
- Python 3.10+ (per pyproject.toml)
- Uses `uv.lock` for reproducible deps