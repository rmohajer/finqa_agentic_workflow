# Weaviate Vector DB Design - Grill Session Notes

## Project Context
- **Goal**: Create a Weaviate Cloud vector database for the ConvFinQA example dataset
- **Source Data**: `data/convfinqa_full.csv` (generated from `scratch/example_data_parsing_experiment.ipynb`)
- **Collection Name**: `ExampleFinQA`
- **Embeddings**: Pre-computed using NVIDIA Nemotron 3.5 Embed 1B (from notebook)
- **Exposure**: Python wrapper class + FastAPI REST endpoints

---

## Decisions Made (via Grill-Me Interview)

### 1. Weaviate Setup
**Decision**: Use Weaviate Cloud with existing credentials
- User has URL and API key ready
- Will store in `.env` file at project root

### 2. Vector Fields
**Decision**: Use pre-computed embeddings from notebook (`summary_emb` column)
- Embeddings generated via NVIDIA Nemotron 3.5 Embed 1B
- 3,458 records with 768-dim vectors (assumed)
- Avoids re-computation, ensures consistency with notebook results

### 3. Schema Design
**Decision**: Single collection `ExampleFinQA` with flat structure

| Property | Type | Description |
|----------|------|-------------|
| `id` | text (UUID) | Original record ID (e.g., "Single_JKHY/2009/page_28.pdf-3") |
| `summary` | text | LLM-generated summary (topic + description + key concepts) |
| `summary_emb` | vector (float[]) | Pre-computed 768-dim embedding |
| `conv_questions` | text[] | Array of conversation questions |
| `conv_answers` | text[] | Array of conversation answers |
| `turn_program` | text[] | Array of program/execution steps |
| `executed_answers` | number[] | Array of numeric executed answers |
| `qa_split` | boolean[] | Array indicating question split types |
| `num_dialogue_turns` | int | Number of dialogue turns (4-7) |
| `has_type2_question` | boolean | Whether example has type-2 questions |
| `has_duplicate_columns` | boolean | Table duplicate columns flag |
| `has_non_numeric_values` | boolean | Table non-numeric values flag |
| `full_example` | text | Full JSON of doc + dialogue + features |

**Vector Configuration**:
- Use `summary_emb` as the vector (pre-computed)
- No vectorizer needed - we provide vectors directly
- Enable hybrid search (BM25 + vector) on `summary` text

### 4. Endpoint Exposure
**Decision**: Dual approach - Python wrapper + FastAPI REST

#### Python Wrapper (`WeaviateClient` class)
```
Methods:
- connect() / close()
- create_collection()
- ingest_dataframe(df: pd.DataFrame) -> int (count inserted)
- search(query: str, top_k: int=3, filters: dict=None) -> List[Dict]
- get_by_id(uuid: str) -> Dict
- delete_collection()
- health_check() -> bool
```

#### FastAPI REST API
```
POST   /search          - Vector/hybrid search
GET    /health          - Health check
POST   /ingest          - Batch ingest (optional)
GET    /collection/info - Collection metadata
```

### 5. Test Coverage
**Decision**: Comprehensive unit tests for every method
- Connection tests
- Schema creation/deletion
- Data ingestion (batch, error handling)
- Search accuracy (known queries with expected results)
- Metadata filtering (by has_type2_question, num_dialogue_turns, etc.)
- API endpoint tests (FastAPI TestClient)
- Edge cases (empty results, invalid filters, timeouts)

### 6. Credentials Management
**Decision**: `.env` file in project root
```
WEAVIATE_URL=https://<cluster>.weaviate.cloud
WEAVIATE_API_KEY=<api-key>
```
Loaded via `python-dotenv` (consistent with project pattern)

### 7. Data Path Configuration
**Decision**: Configurable via `config/config-common-yaml`
```yaml
CONVFINQA_DATA_PATH: "data/convfinqa_full.csv"
WEAVIATE_COLLECTION_NAME: "ExampleFinQA"
```

### 8. Vectorization Strategy
**Decision**: Import pre-computed vectors directly
- Use `summary_emb` from CSV as vector property
- No Weaviate vectorizer module needed
- Faster ingestion, guaranteed consistency with notebook

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ConvFinQA Dataset                        │
│  data/convfinqa_full.csv (3,458 rows)                      │
│  - summary (text)                                           │
│  - summary_emb (768-dim vector)                             │
│  - metadata fields                                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              WeaviateClient (Python Wrapper)                │
│  - Connection management                                    │
│  - Schema management                                        │
│  - Batch ingestion with vectors                             │
│  - Search (vector + hybrid + filters)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌───────────────────────┐     ┌───────────────────────┐
│  LangGraph Agent      │     │  FastAPI REST API     │
│  (agentic_system.py)  │     │  (/search, /health)   │
│  - find_examples()    │     │  - External access    │
└───────────────────────┘     └───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Weaviate Cloud                           │
│  Collection: ExampleFinQA                                   │
│  - 3,458 objects                                            │
│  - Vector index (HNSW) on summary_emb                       │
│  - Inverted index on summary (BM25)                         │
│  - Property indexes on filterable fields                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Replace `find_relevant_document_fast()` in `agentic_system.py`
Current implementation uses numpy cosine similarity on in-memory DataFrame.
New implementation will use WeaviateClient.search()

### 2. Configuration Loading
Extend existing config loader in `tool_helpers.py` / `agentic_system.py` to read Weaviate settings

### 3. Startup Initialization
- Verify collection exists on app startup
- Optionally re-ingest if data changed (checksum comparison)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Weaviate Cloud rate limits | Batch ingestion (100 objects/batch), retry with backoff |
| Vector dimension mismatch | Validate embedding dim matches collection config (768) |
| Credential exposure | `.env` in .gitignore, never commit |
| Data drift (CSV updated) | Checksum-based re-ingestion trigger |
| Search latency | Use hybrid search, limit top_k, enable Weaviate caching |

---

## Next Steps (Execution Order)

1. **Config & Credentials**
   - Update `config/config-common-yaml`
   - Create `.env.example` template
   - User adds actual credentials to `.env`

2. **Notebook Extension**
   - Add Weaviate ingestion cells to `example_data_parsing_experiment.ipynb`
   - Test ingestion with subset of data

3. **Python Wrapper**
   - Create `src/vector_db/weaviate_client.py`
   - Unit tests in `tests/vector_db_tests.py`

4. **FastAPI Endpoint**
   - Create `src/vector_db/api.py`
   - Integration tests

5. **Agent Integration**
   - Update `agentic_system.py` to use WeaviateClient
   - Replace in-memory search

6. **End-to-End Testing**
   - Full pipeline test
   - Performance benchmarks