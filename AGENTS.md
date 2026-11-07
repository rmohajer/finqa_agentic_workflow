# AGENTS.md - FinQA Agentic Workflow

## Quick Start
```bash
export NVIDIA_API_KEY="your_key"  # Required for LLM
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp-key.json"  # For Document AI (optional)
uv sync  # or pip install -e ".[test]"
python app.py  # Launches Gradio on localhost:7860 (share=False)
```

## Architecture Essentials
- **Entry point**: `app.py` → Gradio UI + async handlers
- **Agent**: `agentic_system.py` → LangGraph ReAct agent (`finqa_graph`) + `FinQAResponseAgent` wrapper
- **State**: `FinState` (extends `MessagesState`) tracks report_state, user_query[], call_mode, context_content, context_examples
- **Two retrieval systems**: User PDF chunks (`df_docs.content_embed`) + ConvFinQA examples (`df_finqa.summary_emb`)
- **Structured output**: Enforced via `model_call_tool` (Pydantic schema: Thought[], References[], Final_answer)
- **Two PDF parsers**: Docling (local, CPU) + Google Document AI (cloud, Form Parser)

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
5. **Hardcoded model** — `nvidia/nemotron-3-ultra-550b-a55b` in agentic_system.py:35

## File Responsibilities
| File | Purpose |
|------|---------|
| `app.py` | Gradio UI, PDF upload, chat handlers, parser dropdown |
| `agentic_system.py` | LangGraph nodes, FinState, FinQAResponseAgent |
| `prompt_helpers.py` | System/user prompt templates |
| `tool_helpers.py` | Math tools + `model_call_tool` output schema |
| `parsing_helpers.py` | Docling PDF→MD, Google Document AI, MarkdownHeaderTextSplitter, embeddings, similarity search |
| `data/convfinqa_filtered.csv` | Few-shot examples (loaded at startup) |
| `tests/parsing_helper_tests.py` | Unit tests for parsing_helpers.py |
| `tests.py` | Main test runner CLI |

## Modification Entry Points
- Add tools → `tool_helpers.py`
- Change prompts → `prompt_helpers.py`
- Modify graph → `agentic_system.py` (StateGraph builder)
- Adjust parsing → `parsing_helpers.py`
- UI changes → `app.py`
- Add tests → `tests/parsing_helper_tests.py`


## Agent Permissions
- **MUST ask before any `edit`, `write`, or `patch` tool call**
- Present: file path, exact change (oldString → newString), reason
- Wait for explicit user confirmation ("yes", "proceed", "go ahead")
- No silent edits — even for "obvious" fixes

## Environment
- `NVIDIA_API_KEY` — Required for LLM (Nemotron 3 Ultra)
- `GOOGLE_CLOUD_PROJECT` — GCP project ID for Document AI
- `DOC_AI_PROCESSOR_ID` — Document AI processor ID (Form Parser)
- `DOC_AI_LOCATION` — Processor location (e.g., `australia-southeast1`)
- `GOOGLE_APPLICATION_CREDENTIALS` — Path to service account key JSON
- Python 3.10+ (per pyproject.toml)
- Uses `uv.lock` for reproducible deps