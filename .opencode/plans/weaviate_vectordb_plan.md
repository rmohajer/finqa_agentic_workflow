# Weaviate Vector DB Implementation Plan

## Overview
Create a Weaviate Cloud vector database for the ConvFinQA dataset with Python wrapper and FastAPI REST endpoints.

## Phase 1: Configuration & Setup

### 1.1 Update Configuration File
- **File**: `config/config-common-yaml`
- **Add**:
  ```yaml
  CONVFINQA_DATA_PATH: "data/convfinqa_full.csv"
  WEAVIATE_COLLECTION_NAME: "ExampleFinQA"
  WEAVIATE_BATCH_SIZE: 100
  ```

### 1.2 Create Environment Template
- **File**: `.env.example`
- **Content**:
  ```
  WEAVIATE_URL=https://your-cluster.weaviate.cloud
  WEAVIATE_API_KEY=your-api-key
  ```

### 1.3 Verify Data File Exists
- **Path**: `data/convfinqa_full.csv`
- **Expected**: 3,458 rows with `summary_emb` column

---

## Phase 2: Notebook Extension

### 2.1 Add Weaviate Ingestion Cells
- **File**: `scratch/example_data_parsing_experiment.ipynb`
- **New Cells**:
  1. Load config & credentials
  2. Initialize Weaviate client
  3. Create collection schema
  4. Batch ingest with vectors
  5. Verify with test queries

---

## Phase 3: Python Wrapper (`src/vector_db/weaviate_client.py`)

### 3.1 Class Structure
```python
class WeaviateClient:
    def __init__(self, url: str, api_key: str, collection_name: str)
    def connect(self) -> bool
    def close(self) -> None
    def create_collection(self) -> bool
    def delete_collection(self) -> bool
    def ingest_dataframe(self, df: pd.DataFrame) -> int
    def search(self, query: str, top_k: int = 3, filters: dict = None) -> List[Dict]
    def get_by_id(self, uuid: str) -> Optional[Dict]
    def health_check(self) -> bool
```

### 3.2 Key Implementation Details
- Use `weaviate-client` v4+ API
- Batch ingestion: 100 objects per batch
- Vector property: `summary_emb` (pre-computed)
- Hybrid search: `alpha=0.75` (vector-weighted)
- Filters: Weaviate `where` clause on properties

---

## Phase 4: FastAPI REST API (`src/vector_db/api.py`)

### 4.1 Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/search` | Vector/hybrid search |
| GET | `/collection/info` | Collection metadata |
| POST | `/ingest` | Batch ingest (optional) |

### 4.2 Request/Response Models
```python
# Search Request
{
  "query": "string",
  "top_k": 3,
  "filters": {"has_type2_question": true},
  "hybrid": true
}

# Search Response
{
  "results": [
    {
      "uuid": "string",
      "summary": "string",
      "score": 0.95,
      "metadata": {...}
    }
  ],
  "took_ms": 42
}
```

---

## Phase 5: Unit Tests (`tests/vector_db_tests.py`)

### 5.1 Test Classes
```python
class TestWeaviateConnection:
    def test_connect_success()
    def test_connect_failure()
    def test_health_check()

class TestCollectionManagement:
    def test_create_collection()
    def test_delete_collection()
    def test_collection_exists()

class TestDataIngestion:
    def test_ingest_batch()
    def test_ingest_with_vectors()
    def test_ingest_empty_dataframe()
    def test_ingest_duplicate_handling()

class TestSearch:
    def test_vector_search()
    def test_hybrid_search()
    def test_filter_by_type2_question()
    def test_filter_by_dialogue_turns()
    def test_search_empty_results()
    def test_search_known_query()

class TestFastAPIEndpoints:
    def test_health_endpoint()
    def test_search_endpoint()
    def test_search_with_filters()
```

### 5.2 Test Fixtures
- Mock Weaviate client for unit tests
- Integration test with real Weaviate (marked `@pytest.mark.integration`)
- Sample data fixture from `convfinqa_full.csv` (first 10 rows)

---

## Phase 6: Agent Integration

### 6.1 Update `agentic_system.py`
- Import `WeaviateClient`
- Initialize in `FinQAResponseAgent.__init__()`
- Replace `find_relevant_document_fast()` with `weaviate_client.search()`

### 6.2 Configuration Loading
- Extend `load_config()` in `tool_helpers.py` to read Weaviate settings
- Pass config to agent initialization

---

## Phase 7: End-to-End Testing

### 7.1 Test Scenarios
1. **Cold start**: App starts, connects to Weaviate, serves queries
2. **Search accuracy**: Known queries return expected examples
3. **Filtering**: Metadata filters work correctly
4. **Performance**: Search latency < 200ms for top-3
5. **Concurrent**: Multiple simultaneous searches

### 7.2 Benchmarks
- Ingestion time for 3,458 records
- Search latency (p50, p95, p99)
- Memory usage

---

## Deliverables Checklist

- [ ] `config/config-common-yaml` updated
- [ ] `.env.example` created
- [ ] Notebook extended with ingestion cells
- [ ] `src/vector_db/weaviate_client.py` implemented
- [ ] `src/vector_db/api.py` implemented
- [ ] `tests/vector_db_tests.py` with full coverage
- [ ] `agentic_system.py` integrated
- [ ] Documentation updated
- [ ] All tests passing

---

## Dependencies

### New Python Packages
```toml
weaviate-client = "^4.0"
fastapi = "^0.109"
uvicorn = "^0.27"
pydantic = "^2.6"
python-dotenv = "^1.0"  # already in project
pytest = "^7.4"  # already in project
pytest-asyncio = "^0.23"
httpx = "^0.26"  # for FastAPI TestClient
```

### Installation
```bash
uv add weaviate-client fastapi uvicorn pydantic httpx pytest-asyncio
```

---

## Timeline Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| 1: Config & Setup | Low | 30 min |
| 2: Notebook Extension | Medium | 1 hr |
| 3: Python Wrapper | High | 2-3 hrs |
| 4: FastAPI API | Medium | 1-2 hrs |
| 5: Unit Tests | High | 2-3 hrs |
| 6: Agent Integration | Medium | 1 hr |
| 7: E2E Testing | Medium | 1 hr |
| **Total** | | **8-11 hrs** |

---

## Success Criteria

1. All unit tests pass (`pytest tests/vector_db_tests.py -v`)
2. Integration test passes with real Weaviate Cloud
3. Search returns relevant examples for known queries
4. Latency < 200ms for top-3 search
5. Agent correctly retrieves examples via Weaviate
6. No regression in existing functionality