#%% !/usr/bin/env python3
"""
Test script for Google Document AI Form Parser and NVIDIA Embeddings.

This tests the GCP Document AI implementation and NVIDIA embedding model
added to parsing_helpers.py and agentic_system.py.
Requires: uv pip install google-cloud-documentai google-cloud-storage python-dotenv langchain-nvidia-ai-endpoints
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Auto-set GOOGLE_APPLICATION_CREDENTIALS if not set but key file exists
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    default_key_path = os.path.expanduser("~/.config/gcp/docai-key.json")
    if os.path.exists(default_key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_key_path
        print(f"🔑 Auto-set GOOGLE_APPLICATION_CREDENTIALS = {default_key_path}")


def test_doc_ai_imports():
    """Test that Document AI imports work."""
    print("Testing Document AI imports...")
    try:
        from google.cloud import documentai
        from google.api_core.client_options import ClientOptions
        print("✅ google-cloud-documentai imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: uv pip install google-cloud-documentai google-cloud-storage")
        return False


def test_env_vars():
    """Test that required environment variables are set."""
    print("\nTesting environment variables...")
    required = [
        "GOOGLE_CLOUD_PROJECT",
        "DOC_AI_PROCESSOR_ID",
        "DOC_AI_LOCATION",
    ]
    
    missing = []
    for var in required:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var} = {value}")
        else:
            print(f"  ❌ {var} = NOT SET")
            missing.append(var)
    
    # Check credentials
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds:
        print(f"  ✅ GOOGLE_APPLICATION_CREDENTIALS = {creds}")
        if not os.path.exists(os.path.expanduser(creds)):
            print(f"     ⚠️  File does not exist!")
    else:
        print(f"  ❌ GOOGLE_APPLICATION_CREDENTIALS = NOT SET")
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    
    return len(missing) == 0


def test_nvidia_embedding_imports():
    """Test that NVIDIA embedding imports work."""
    print("Testing NVIDIA embedding imports...")
    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        print("✅ langchain-nvidia-ai-endpoints imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: uv pip install langchain-nvidia-ai-endpoints")
        return False


def test_nvidia_embedding_env_vars():
    """Test that NVIDIA embedding environment variables are set."""
    print("\nTesting NVIDIA embedding environment variables...")
    required = [
        "NVIDIA_EMBEDDING_API_KEY",
    ]
    
    missing = []
    for var in required:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var} = {value[:20]}...")
        else:
            print(f"  ❌ {var} = NOT SET")
            missing.append(var)
    
    return len(missing) == 0


def test_nvidia_embedding_model():
    """Test the NVIDIA embedding model (Nemotron 3.5 Embed 1B)."""
    print("\nTesting NVIDIA embedding model...")
    
    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        print("✅ NVIDIAEmbeddings imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Get model name from config
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config-common-yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get("EMBEDDING_MODEL_NAME", "nvidia/nemotron-3.5-embed-1b")
        print(f"  Using model: {model_name}")
    except Exception as e:
        model_name = "nvidia/nemotron-3.5-embed-1b"
        print(f"  Using default model: {model_name}")
    
    api_key = os.getenv("NVIDIA_EMBEDDING_API_KEY")
    if not api_key:
        print("  ❌ NVIDIA_EMBEDDING_API_KEY not set")
        return False
    
    try:
        print("  Initializing NVIDIAEmbeddings...")
        embedding_model = NVIDIAEmbeddings(
            model=model_name,
            api_key=api_key,
            truncate="END",
        )
        print("  ✅ Embedding model initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize embedding model: {e}")
        return False
    
    # Test embedding a simple text
    test_text = "This is a test sentence for embedding."
    try:
        print(f"  Embedding test text: '{test_text}'")
        embedding = embedding_model.embed_query(test_text)
        print(f"  ✅ Embedding generated successfully")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        
        # Test embedding multiple texts (batch)
        test_texts = [
            "First test document.",
            "Second test document with different content.",
            "Third document about financial reports."
        ]
        try:
            print(f"  Testing batch embedding with embed_documents...")
            embeddings = embedding_model.embed_documents(test_texts)
            print(f"  ✅ Batch embedding successful: {len(embeddings)} embeddings")
            print(f"  Each embedding dimension: {len(embeddings[0])}")
        except Exception as batch_error:
            print(f"  ⚠️  Batch embedding failed (server error): {batch_error}")
            print(f"  Falling back to individual embed_query calls...")
            embeddings = []
            for i, text in enumerate(test_texts):
                emb = embedding_model.embed_query(text)
                embeddings.append(emb)
                print(f"    Text {i+1}: dimension {len(emb)}")
            print(f"  ✅ Individual embeddings successful: {len(embeddings)} embeddings")
            print(f"  Each embedding dimension: {len(embeddings[0])}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error during embedding: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nvidia_embedding_with_fin_doc_parser(pdf_path: str):
    """Test the full fin_doc_parser with NVIDIA embedding model."""
    print(f"\nTesting fin_doc_parser with NVIDIA embedding model...")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None, None
    
    try:
        from src.parsing_helpers import fin_doc_parser
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config-common-yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get("EMBEDDING_MODEL_NAME", "nvidia/nemotron-3.5-embed-1b")
        min_chunk_length = config.get("MIN_CHUNK_LENGTH", 800)
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None, None
    
    # Load NVIDIA embedding model
    api_key = os.getenv("NVIDIA_EMBEDDING_API_KEY")
    if not api_key:
        print("  ❌ NVIDIA_EMBEDDING_API_KEY not set")
        return None, None
    
    try:
        print(f"  Loading NVIDIA embedding model ({model_name})...")
        embedding_model = NVIDIAEmbeddings(
            model=model_name,
            api_key=api_key,
            truncate="END",
        )
        print("  ✅ NVIDIA Embedding model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load embedding model: {e}")
        return None, None
    
    try:
        print(f"  Parsing with docling parser...")
        df_docs, docs_with_meta = fin_doc_parser(
            document=pdf_path,
            embedding_model=embedding_model,
            min_chunk_length=min_chunk_length,
            parser="docling"
        )
        print(f"  ✅ Parsing successful!")
        print(f"  Chunks created: {len(df_docs)}")
        print(f"  Columns: {list(df_docs.columns)}")
        if len(df_docs) > 0:
            print(f"  First chunk preview: {df_docs.iloc[0]['content'][:100]}...")
            # Check embedding dimension
            if 'embedding' in df_docs.columns:
                emb = df_docs.iloc[0]['embedding']
                print(f"  Embedding dimension: {len(emb)}")
        return df_docs, docs_with_meta
    except Exception as e:
        print(f"  ❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_parse_with_doc_ai(pdf_path: str):
    """Test the parse_with_doc_ai function from parsing_helpers."""
    print(f"\nTesting parse_with_doc_ai with: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None
    
    try:
        from src.parsing_helpers import parse_with_doc_ai
        print("✅ parse_with_doc_ai imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None
    
    try:
        print("Calling Document AI...")
        markdown = parse_with_doc_ai(pdf_path)
        print("✅ Document AI parsing successful!")
        print(f"Output length: {len(markdown)} characters")
        return markdown
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_fin_doc_parser(pdf_path: str, parser: str = "doc_ai"):
    """Test the full fin_doc_parser with both parsers using NVIDIA embeddings."""
    print(f"\nTesting fin_doc_parser with parser='{parser}' (NVIDIA embeddings)...")
    
    try:
        from src.parsing_helpers import fin_doc_parser, parse_with_doc_ai
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config-common-yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get("EMBEDDING_MODEL_NAME", "nvidia/nemotron-3.5-embed-1b")
        min_chunk_length = config.get("MIN_CHUNK_LENGTH", 800)
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None, None
    
    # Debug: Show raw parser output difference
    print(f"\n  🔍 Debugging parser output for '{parser}':")
    if parser == "doc_ai":
        try:
            raw_markdown = parse_with_doc_ai(pdf_path)
            import re
            headers = re.findall(r'^#{1,3}\s+.+', raw_markdown, re.MULTILINE)
            print(f"    Raw markdown length: {len(raw_markdown)} chars")
            print(f"    Markdown headers (#, ##, ###): {len(headers)}")
            if len(headers) == 0:
                print(f"    ⚠️  NO MARKDOWN HEADERS - MarkdownHeaderTextSplitter will create 1 giant chunk!")
        except Exception as e:
            print(f"    Could not parse raw output: {e}")
    else:
        # For docling, we'd need to run the converter to check
        print(f"    Docling produces proper markdown with headers (handled internally)")
    
    # Load NVIDIA embedding model
    api_key = os.getenv("NVIDIA_EMBEDDING_API_KEY")
    if not api_key:
        print("  ❌ NVIDIA_EMBEDDING_API_KEY not set")
        return None, None
    
    try:
        print(f"  Loading NVIDIA embedding model ({model_name})...")
        embedding_model = NVIDIAEmbeddings(
            model=model_name,
            api_key=api_key,
            truncate="END",
        )
        print("  ✅ NVIDIA Embedding model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load embedding model: {e}")
        return None, None
    
    try:
        print(f"  Parsing with {parser}...")
        df_docs, docs_with_meta = fin_doc_parser(
            document=pdf_path,
            embedding_model=embedding_model,
            min_chunk_length=min_chunk_length,
            parser=parser
        )
        print(f"  ✅ Parsing successful!")
        print(f"  Chunks created: {len(df_docs)}")
        print(f"  Columns: {list(df_docs.columns)}")
        if len(df_docs) > 0:
            print(f"  First chunk preview: {df_docs.iloc[0]['content'][:100]}...")
            print(f"  Chunk sizes: {[len(c) for c in df_docs['content'].tolist()[:5]]}")
            # Check embedding dimension
            if 'embedding' in df_docs.columns:
                emb = df_docs.iloc[0]['embedding']
                print(f"  Embedding dimension: {len(emb)}")
        return df_docs, docs_with_meta
    except Exception as e:
        print(f"  ❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_output(markdown: str, pdf_path: str, parser: str):
    """Save parsed output to file."""
    output_dir = Path("output/gcp_doc_ai")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    output_file = output_dir / f"{pdf_name}_{parser}_output.txt"
    output_file.write_text(markdown)
    print(f"\n📄 Output saved to: {output_file}")


#%%

def main():
    print("=" * 60)
    print("Testing Google Document AI Form Parser & NVIDIA Embeddings")
    print("=" * 60)
    
    # Test NVIDIA Embedding imports
    print("\n" + "=" * 60)
    print("Testing NVIDIA Embedding Imports")
    print("=" * 60)
    if not test_nvidia_embedding_imports():
        print("\n⚠️  NVIDIA embedding imports failed. Skipping embedding tests.")
    else:
        # Test NVIDIA Embedding env vars
        if test_nvidia_embedding_env_vars():
            # Test NVIDIA Embedding model
            test_nvidia_embedding_model()
    
    # Test Document AI imports
    print("\n" + "=" * 60)
    print("Testing Document AI Imports")
    print("=" * 60)
    if not test_doc_ai_imports():
        print("\n⚠️  Document AI imports failed. Skipping Document AI tests.")
    else:
        # Test environment
        if not test_env_vars():
            print("\n⚠️  Some environment variables missing. Check .env file.")
            print("Required: GOOGLE_CLOUD_PROJECT, DOC_AI_PROCESSOR_ID, DOC_AI_LOCATION")
            print("And GOOGLE_APPLICATION_CREDENTIALS pointing to service account key")
        else:
            # List all processors to debug
            print("\n" + "=" * 60)
            print("Listing all Document AI processors in project")
            print("=" * 60)
            try:
                from src.parsing_helpers import list_doc_ai_processors
                list_doc_ai_processors()
            except Exception as e:
                print(f"❌ Failed to list processors: {e}")
    
    # Find test PDF
    test_pdfs = [
        "data/CVs-principal genai scientist.pdf",
    ]
    
    pdf_path = None
    for p in test_pdfs:
        if os.path.exists(p):
            pdf_path = p
            break
    
    if not pdf_path:
        print("\n❌ No test PDF found. Place a PDF in data/ directory.")
        return
    
    print(f"\n📄 Using test PDF: {pdf_path}")
    
    # Test parse_with_doc_ai directly
    print("\n" + "=" * 60)
    print("Testing parse_with_doc_ai directly")
    print("=" * 60)
    markdown = test_parse_with_doc_ai(pdf_path)
    if markdown:
        save_output(markdown, pdf_path, "doc_ai")
        
        # Show preview
        print("\n" + "=" * 60)
        print("PARSED OUTPUT PREVIEW (first 100 chars)")
        print("=" * 60)
        print(markdown[:100])
        print("..." if len(markdown) > 100 else "")
        print("=" * 60)
    
    # Test full pipeline with doc_ai (using NVIDIA embeddings)
    print("\n" + "=" * 60)
    print("Testing full fin_doc_parser pipeline with doc_ai (NVIDIA embeddings)")
    print("=" * 60)
    df_docs, docs = test_fin_doc_parser(pdf_path, parser="doc_ai")
    
    # Also test docling for comparison (using NVIDIA embeddings)
    print("\n" + "=" * 60)
    print("Testing full fin_doc_parser pipeline with docling (NVIDIA embeddings)")
    print("=" * 60)
    df_docs_docling, docs_docling = test_fin_doc_parser(pdf_path, parser="docling")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if df_docs is not None:
        print(f"✅ doc_ai (NVIDIA): {len(df_docs)} chunks")
    else:
        print("❌ doc_ai (NVIDIA): Failed")
    
    if df_docs_docling is not None:
        print(f"✅ docling (NVIDIA): {len(df_docs_docling)} chunks")
    else:
        print("❌ docling (NVIDIA): Failed")



#%%
if __name__ == "__main__":
    main()