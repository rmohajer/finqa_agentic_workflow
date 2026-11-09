# FinQA Agentic Workflow - Project Context

## Project Overview
A Gradio-based chat application that allows users to upload financial PDF reports and ask conversational questions about them. Uses a LangGraph ReAct agent with tool-use capabilities for numerical reasoning over financial documents.

**Key Technologies**: LangGraph, LangChain, NVIDIA Nemotron 3 Ultra (LLM), NVIDIA Nemotron 3.5 Embed 1B (embeddings), Gradio, Docling (PDF parsing), Google Document AI (optional cloud parser)

---

## Architecture

### Core Components

| File | Purpose |
|------|---------|
| `app.py` | Gradio UI + async handlers for PDF upload & chat, parser selection dropdown, session management |
| `agentic_system.py` | LangGraph ReAct agent, state management, FinQAResponseAgent wrapper, lazy embedding initialization |
| `prompt_helpers.py` | System/user prompt templates with structured output format |
| `tool_helpers.py` | Math tools (add, subtract, multiply, divide, sqrt, power) + model_call_tool, config loader |
| `parsing_helpers.py` | PDF parsing (Docling + Google Document AI OCR/Layout), markdown chunking, embedding generation, similarity search |
| `dataset.md` | ConvFinQA dataset documentation |
| `tests/parsing_helper_tests.py` | Unit tests for parsing helpers (81 tests) |
| `tests.py` | Main test runner CLI |
| `config/config-common-yaml` | Centralized configuration (models, chunk length, Document AI settings) |

### Data Flow
1. **PDF Upload** → `fin_doc_parser()` → markdown chunks with metadata → embeddings → `df_docs` (DataFrame)
2. **User Question** → `FinQAResponseAgent.predict()` → LangGraph execution
3. **LangGraph Nodes**:
   - `check_info_node` - validates report exists, detects query continuation
   - `call_model_node` - retrieves relevant docs/examples, invokes LLM with tools
   - `tool_node` - executes tool calls (math operations)
   - Loops until `call_mode == 'user'` (final answer)

### State Schema (`FinState`)
```python
FinState(MessagesState):
    report_state: bool           # PDF loaded?
    user_query: List[str]        # All questions in conversation
    next_node: Optional[str]     # "call_model_node" | "END"
    call_mode: Literal['user','tool_call']
    context_content: str         # Relevant chunks from user's PDF
    context_examples: str        # Relevant ConvFinQA examples
    message_out: Optional[str]   # Final answer
    full_response: Any           # Full LLM response
    query_cont: List[str]        # Connected/continuation queries
```

---

## Key Implementation Details

### LLM Configuration
- **Model**: `nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA API (Nemotron 3 Ultra - frontier reasoning & agentic workflows)
- **Fallback**: `openai/gpt-oss-120b` via Groq API (currently active)
- **Temperature**: 0 (deterministic)
- **Embeddings**: `nvidia/nemotron-3.5-embed-1B` via NVIDIA API (2048-dim)
- **Package**: `langchain-nvidia-ai-endpoints` (`ChatNVIDIA`, `NVIDIAEmbeddings`)

### Prompt Structure
- **System**: Financial expert with tool access, step-by-step reasoning
- **User (new query)**: Chat history + context + examples + tools → structured output
- **User (tool continuation)**: Context + chat history + tools → continue reasoning
- **Output Format**: `Thought: [...]`, `References: [...]`, `Final_answer: ...`

### Tools Available
| Tool | Purpose |
|------|---------|
| `add`, `subtract`, `multiply`, `divide`, `sqrt`, `power` | Basic math operations |
| `model_call_tool` | Structured output (Thought, References, Final_answer) |

### Document Parsing (`parsing_helpers.py`)
- **Docling** converts PDF → Markdown (local, CPU-only, MPS-compatible)
- **Google Document AI** (free tier: 1000 pages/mo):
  - **Enterprise Document OCR** (default): paragraphs, tables, blocks → infers headings for markdown
  - **Layout Parser** (paid): native heading detection, chunking
  - **Form Parser** (legacy): form fields, entities
- **MarkdownHeaderTextSplitter** splits on `#`, `##`, `###` headers
- Metadata merged into content: `[Section: X | Subsection: Y]\n\nContent`
- Small chunks merged until `min_chunk_length` (default 800 chars, configurable)
- Embeddings generated per chunk for similarity search
- **Imageless mode** for Document AI (30 page limit vs 15)
- **Automatic batching** for documents exceeding page limit

### Similarity Search
```python
find_relevant_document_fast(embedding_model, query, df, embed_col, top_k)
# Cosine similarity between query embedding and document embeddings
```

### Conversation Continuity Detection
`check_info_node` uses LLM to classify: "Is new query a continuation of previous queries? Yes/No"
- If Yes: adds to `query_cont` list (used for context retrieval)
- If No: resets `query_cont` to `[new_query]`

### ConvFinQA Embedding Management (Lazy Initialization)
- **Cache file**: `data/convfinqa_embeddings_cache.npy` (avoids recomputing on restart)
- **First run**: ~2-3 minutes (3,892 conversations via batch `embed_documents`)
- **Subsequent runs**: Instant load from `.npy` cache
- **Lazy initialization**: Only triggered on first `call_model_node` execution
- **Called explicitly at app startup** via `_ensure_finqa_embeddings()` in `app.py`

---

## Running the App

```bash
# Install dependencies
uv sync  # or pip install -e ".[test]"

# Set environment variables
export NVIDIA_API_KEY="your_nvidia_api_key_here"  # Starts with nvapi-
export NVIDIA_EMBEDDING_API_KEY="your_nvidia_embedding_key"  # For Nemotron 3.5 Embed 1B
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp-key.json"  # For Document AI (optional)

# Document AI config in .env / config/config-common-yaml:
# GOOGLE_CLOUD_PROJECT=focal-welder-366300
# DOC_AI_PROCESSOR_ID=e1dfbf6e967f9ffe
# DOC_AI_LOCATION=australia-southeast1
# DOC_AI_OCR_PROCESSOR_VERSION=pretrained-ocr-v2.0-2023-06-02
# DOC_AI_PARSER_TYPE=ocr

# Run
python app.py
# Launches on http://localhost:7860 (share=False)
```

---

## Dataset (ConvFinQA)
- **Source**: Cleaned version of [ConvFinQA](https://github.com/czyssrs/ConvFinQA)
- **Size**: 3,892 conversations, 14,115 questions
- **Types**: Type I (simple, single question decomposed), Type II (hybrid, two related questions)
- **Used for**: Few-shot examples in `context_examples` (top-2 most similar conversations)
- **Loaded at startup**: `df_finqa` with pre-computed `summary_emb` embeddings (2048-dim Nemotron)
- **Data prep notebook**: `scratch/data_parsing_experiment.ipynb` (summaries, embeddings, restructuring)

---

## Data Preparation Pipeline (`scratch/data_parsing_experiment.ipynb`)

The notebook documents the full pipeline for preparing ConvFinQA data:

1. **Load raw dataset** → JSON → DataFrame (train + dev splits)
2. **Create `full_example`** → JSON combining `doc`, `dialogue`, `features`
3. **Summarize documents** → Using Groq (llama-3.1-8b-instant) and/or NVIDIA (Nemotron 3.5 Bolt E4B) with retry mechanism (3 retries, 5s delay on 502/timeout)
4. **Add embeddings** → Nemotron 3.5 Embed 1B with retry on 502 Bad Gateway (3 retries, 5s delay)
5. **Restructure dataset**:
   - Parse `dialogue` JSON → `dialogue_dict` columns
   - Create `conv_text_full` → formatted Q/R/A text for few-shot
   - Parse `features` JSON → individual columns (`has_type2_question`, etc.)
6. **Search function** → `find_relevant_document_fast()` with cosine similarity

---

## Environment Variables Required
| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | LLM access (Nemotron 3 Ultra via `ChatNVIDIA`) |
| `NVIDIA_EMBEDDING_API_KEY` | Embeddings (Nemotron 3.5 Embed 1B via `NVIDIAEmbeddings`) |
| `GROQ_API_KEY` | Fallback LLM (Groq `ChatGroq`) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for Document AI |
| `DOC_AI_PROCESSOR_ID` | Document AI processor ID |
| `DOC_AI_LOCATION` | Processor location (e.g., `australia-southeast1`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key JSON |

All config centralized in `config/config-common-yaml`.

---

## Known Issues / TODOs
1. **Global `df_docs` mutation** in `FinQAResponseAgent.predict()` - sets `globals()["df_docs"]` - not thread-safe for multi-user
2. **Cache cleaning** on startup (app.py lines 18-27) - aggressive, may cause issues in production
3. **No persistence** - InMemorySaver loses state on restart
4. **Single global agent** - `FinQAAgent` shared across all users (uses `thread_id` for isolation)
5. **Hardcoded model** - `NVIDIA_MODEL` in config should be the single source of truth

---

## Testing the Agent
```python
# Quick test in Python
from agentic_system import FinQAAgent, finqa_graph
import asyncio

agent = FinQAResponseAgent(finqa_graph)
agent.set_report_state(True)
# agent.set_documents(df_docs)  # after parsing a PDF

result = await agent.predict({"user_id": "test", "user_input": "What is the revenue?"})
print(result['output'])
```

### Running Unit Tests
```bash
# Run parsing helper tests
uv run python tests.py parsing
# or
uv run python -m pytest tests/parsing_helper_tests.py -v

# Run all tests
uv run python tests.py all

# Run with coverage
uv run python tests.py coverage
```

---

## File Structure
```
finqa_app/
├── app.py                    # Gradio entry point + session management
├── agentic_system.py         # LangGraph agent + FinQAResponseAgent
├── prompt_helpers.py         # Prompt templates
├── tool_helpers.py           # Math tools + output schema tool + config loader
├── parsing_helpers.py        # PDF parsing, chunking, embeddings, search
├── config/
│   └── config-common-yaml    # Centralized configuration
├── data/
│   ├── convfinqa_filtered.csv      # Few-shot examples (loaded at startup)
│   ├── convfinqa_embeddings_cache.npy  # Cached ConvFinQA embeddings
│   └── convfinqa_dataset.json      # Raw dataset
├── scratch/
│   ├── data_parsing_experiment.ipynb  # Data prep pipeline
│   └── test_parsers.py           # Integration tests for parsers
├── tests/
│   ├── parsing_helper_tests.py  # 81 unit tests
│   └── conftest.py              # Shared fixtures
├── tests.py                  # Test runner CLI
├── pyproject.toml
├── uv.lock
├── .env                      # Environment variables
└── OPENCODE.md / AGENTS.md   # This documentation
```

---

## Useful Commands

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy .

# Run tests
uv run python tests.py parsing
# or
pytest tests/parsing_helper_tests.py -v

# Run with coverage
uv run python tests.py coverage
```

---

## Context for Future Sessions

When resuming work on this project:
1. The agent uses a **ReAct pattern** with LangGraph - model calls tools, tools return, model continues
2. **Two retrieval systems**: User's PDF chunks (`df_docs.content_embed`) + ConvFinQA examples (`df_finqa.summary_emb`)
3. **Structured output** enforced via `model_call_tool` with Pydantic schema
4. **Conversation memory** via LangGraph's `InMemorySaver` keyed by `thread_id` (user_id)
5. **Query continuation detection** enables multi-turn reasoning over same topic
6. **Two PDF parsers available**: Docling (local) and Google Document AI (cloud OCR/Layout)
7. **ConvFinQA embeddings cached** in `.npy` file - lazy init on first use, pre-computed at app startup
8. **Retry mechanisms** for 502 Bad Gateway and timeouts (3 retries, 5s delay) in notebook and embedding calls
9. **Unit tests** in `tests/parsing_helper_tests.py` with 81+ passing tests

Key entry points for modifications:
- Add tools → `tool_helpers.py`
- Change prompts → `prompt_helpers.py`
- Modify graph structure → `agentic_system.py` (builder)
- Adjust parsing → `parsing_helpers.py`
- UI changes → `app.py`
- Add tests → `tests/parsing_helper_tests.py`
- Config changes → `config/config-common-yaml`