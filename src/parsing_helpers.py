import os
import yaml
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# from langchain_mistralai import MistralAIEmbeddings
import pandas as pd
import json
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from ast import literal_eval
from joblib import Parallel, delayed
import math
import time
import re
import numpy as np
from numpy.linalg import norm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from collections import defaultdict
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions

# Google Document AI imports
try:
    from google.cloud import documentai
    from google.api_core.client_options import ClientOptions
    DOC_AI_AVAILABLE = True
except ImportError:
    DOC_AI_AVAILABLE = False


# Load configuration from YAML file
def load_config():
    """Load configuration from config-common-yaml file."""
    config_path = Path(__file__).parent.parent / "config" / "config-common-yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()


def parse_with_doc_ai(
    pdf_path: str,
    field_mask: str = "text,pages.pageNumber,pages.paragraphs,pages.tables,pages.blocks",
    processor_version_id: str = None,
    page_numbers: list = None,
) -> str:
    """
    Parse a PDF using Google Document AI Enterprise Document OCR (free tier: 1000 pages/month) 
    and return markdown-formatted text.
    
    OCR Processor provides (free tier):
    - Text extraction with paragraphs, blocks, lines, tokens
    - Table detection
    - Bounding boxes for layout analysis
    - Font/style information (in newer versions)
    
    Args:
        pdf_path: Path to the PDF file to parse.
        field_mask: Comma-separated list of fields to return.
        processor_version_id: Optional specific processor version to use.
        page_numbers: Optional list of page numbers to process (1-indexed).
        
    Returns:
        Markdown-formatted text with inferred heading structure from paragraph layout.
        
    Raises:
        ImportError: If google-cloud-documentai is not installed.
        ValueError: If required environment variables are not set.
        Exception: For API errors.
    """
    if not DOC_AI_AVAILABLE:
        raise ImportError("google-cloud-documentai not installed. Run: pip install google-cloud-documentai")
    
    # Load environment variables
    load_dotenv()
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    # Use OCR processor ID from config (free tier)
    processor_id = config.get("DOC_AI_OCR_PROCESSOR_ID", os.getenv("DOC_AI_PROCESSOR_ID"))
    location = os.getenv("DOC_AI_LOCATION", "australia-southeast1")
    
    # Default to stable OCR version
    if processor_version_id is None:
        processor_version_id = config.get("DOC_AI_OCR_PROCESSOR_VERSION", "pretrained-ocr-v2.0-2023-06-02")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in environment")
    if not processor_id:
        raise ValueError("DOC_AI_OCR_PROCESSOR_ID not set in environment")
    
    # Initialize client
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    
    # Build the processor resource name with version
    name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    
    print(f"🔍 Using Enterprise Document OCR (free tier): {name}")
    print(f"🔍 API endpoint: {location}-documentai.googleapis.com")
    
    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    file_size_mb = len(pdf_content) / (1024 * 1024)
    print(f"📄 PDF size: {file_size_mb:.2f} MB")
    
    # Load binary data
    raw_document = documentai.RawDocument(content=pdf_content, mime_type="application/pdf")
    
    # Configure process options
    process_options = documentai.ProcessOptions()
    
    if page_numbers:
        process_options.individual_page_selector = documentai.ProcessOptions.IndividualPageSelector(
            pages=page_numbers
        )
    
    # Configure the process request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
        field_mask=field_mask,
        process_options=process_options,
    )
    
    # Process the document
    try:
        print("📤 Sending request to Document AI OCR...")
        result = client.process_document(request=request)
        document = result.document
    except Exception as e:
        # Check if it's a processor version not found error
        if "PROCESSOR_VERSION_NOT_FOUND" in str(e) or "ProcessorVersion with id" in str(e):
            print(f"⚠️  OCR version '{processor_version_id}' not found on this processor.")
            print(f"⚠️  Trying default processor (no version)...")
            # Try without version
            name = client.processor_path(project_id, location, processor_id)
            request = documentai.ProcessRequest(
                name=name,
                raw_document=raw_document,
                field_mask=field_mask,
                process_options=process_options,
            )
            result = client.process_document(request=request)
            document = result.document
        elif "PAGE_LIMIT_EXCEEDED" in str(e) and page_numbers is None:
            print("📄 Document exceeds page limit. Processing in batches...")
            return _process_document_in_batches(
                client, name, raw_document, field_mask, process_options
            )
        else:
            raise
    
    print(f"✅ Document processed: {len(document.pages)} pages, {len(document.text)} chars")
    
    # Count paragraphs and tables
    total_paragraphs = sum(len(page.paragraphs) for page in document.pages)
    total_tables = sum(len(page.tables) for page in document.pages)
    print(f"📋 Found {total_paragraphs} paragraphs, {total_tables} tables")
    
    # Convert to markdown using OCR paragraph structure
    return _ocr_document_to_markdown(document)



def _process_document_in_batches(
    client,
    processor_name: str,
    raw_document,
    field_mask: str,
    process_options,
    batch_size: int = 15
) -> str:
    """
    Process a large document in batches to avoid page limit.
    
    Args:
        client: DocumentProcessorServiceClient
        processor_name: Full processor resource name
        raw_document: RawDocument object
        field_mask: Field mask for the request
        process_options: ProcessOptions object
        batch_size: Number of pages per batch (default 15)
        
    Returns:
        Combined markdown from all batches
    """
    from google.cloud import documentai
    
    all_markdown_parts = []
    page_num = 1
    
    while True:
        # Create page selector for this batch
        batch_pages = list(range(page_num, page_num + batch_size))
        batch_process_options = documentai.ProcessOptions(
            individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
                pages=batch_pages
            )
        )
        
        # Copy other process options (like imageless_mode)
        if getattr(process_options, 'imageless_mode', False):
            batch_process_options.imageless_mode = True
        
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
            field_mask=field_mask,
            process_options=batch_process_options,
        )
        
        print(f"📤 Processing batch: pages {batch_pages[0]}-{batch_pages[-1]}")
        result = client.process_document(request=request)
        document = result.document
        
        if not document.pages:
            break
            
        markdown = _doc_ai_to_markdown(document)
        all_markdown_parts.append(markdown)
        
        print(f"✅ Batch processed: {len(document.pages)} pages")
        
        if len(document.pages) < batch_size:
            break
            
        page_num += batch_size
    
    combined = "\n\n---\n\n".join(all_markdown_parts)
    print(f"✅ All batches complete. Total pages processed.")
    
    return combined


def _doc_ai_to_markdown(document) -> str:
    """
    Convert Google Document AI Document object to markdown-formatted string.
    Preserves text, tables (as markdown tables), and key-value pairs.
    """
    markdown_parts = []
    
    # Add full text
    if document.text:
        markdown_parts.append(document.text)
    
    # Process pages for tables and layout
    for page in document.pages:
        page_text = []
        
        # Extract tables from this page
        for table in page.tables:
            table_md = _table_to_markdown(table, document.text)
            if table_md:
                page_text.append(f"\n{table_md}\n")
        
        # Extract form fields (key-value pairs)
        if page.form_fields:
            page_text.append("\n--- Form Fields ---")
            for field in page.form_fields:
                field_name = _get_text(field.field_name, document.text)
                field_value = _get_text(field.field_value, document.text)
                if field_name and field_value:
                    page_text.append(f"**{field_name}**: {field_value}")
        
        if page_text:
            markdown_parts.append("\n".join(page_text))
    
    # Also extract entities if available
    if document.entities:
        markdown_parts.append("\n--- Entities ---")
        for entity in document.entities:
            entity_type = entity.type_
            entity_text = _get_text(entity.mention_text, document.text) if entity.mention_text else ""
            confidence = entity.confidence
            if entity_text:
                markdown_parts.append(f"**{entity_type}** ({confidence:.2f}): {entity_text}")
    
    return "\n\n".join(markdown_parts)




def _find_and_convert_table(block, document) -> str:
    """Find table in document pages that matches the layout block and convert to markdown."""
    if not block.layout or not block.layout.text_anchor:
        return None
    
    # Get the text range of the block
    block_start = None
    block_end = None
    for segment in block.layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index) if segment.end_index else len(document.text)
        if block_start is None or start < block_start:
            block_start = start
        if block_end is None or end > block_end:
            block_end = end
    
    # Search for tables in pages that overlap with this block
    for page in document.pages:
        for table in page.tables:
            # Get table text range
            table_start = None
            table_end = None
            if table.layout and table.layout.text_anchor:
                for segment in table.layout.text_anchor.text_segments:
                    start = int(segment.start_index) if segment.start_index else 0
                    end = int(segment.end_index) if segment.end_index else len(document.text)
                    if table_start is None or start < table_start:
                        table_start = start
                    if table_end is None or end > table_end:
                        table_end = end
                
                # Check if table overlaps with block
                if table_start is not None and table_end is not None:
                    if table_start >= block_start and table_end <= block_end:
                        return _table_to_markdown(table, document.text)
    
    return None


def _ocr_document_to_markdown(document) -> str:
    """
    Convert Google Document AI OCR Document object to markdown-formatted string.
    
    Uses page.paragraphs which provides structured text with layout info.
    Infers headings based on:
    - Font size (larger = heading)
    - Position (top of page, isolated)
    - Text patterns (ALL CAPS, short lines)
    - Bold/italic detection from font info
    
    Also extracts tables from page.tables.
    """
    markdown_parts = []
    
    for page_idx, page in enumerate(document.pages):
        page_num = page.page_number if page.page_number else page_idx + 1
        markdown_parts.append(f"\n<!-- Page {page_num} -->\n")
        
        # Get all paragraphs for this page
        paragraphs = list(page.paragraphs) if page.paragraphs else []
        
        if not paragraphs:
            # Fallback: use full page text
            if page.layout and page.layout.text_anchor:
                page_text = _get_text(page.layout, document.text)
                if page_text.strip():
                    markdown_parts.append(page_text.strip())
            continue
        
        # Analyze paragraphs to infer structure
        # Calculate average font size for heading detection
        font_sizes = []
        for para in paragraphs:
            if para.layout and para.layout.text_anchor:
                # Get font size from first token if available
                # Note: OCR may not always provide font info
                pass
        
        # Simple heuristic: first paragraph on page, short, ALL CAPS or Title Case = heading
        for i, para in enumerate(paragraphs):
            if not para.layout or not para.layout.text_anchor:
                continue
                
            para_text = _get_text(para.layout, document.text).strip()
            if not para_text:
                continue
            
            # Check if this might be a heading
            is_heading = False
            heading_level = 0
            
            # Heuristic 1: Short line, ALL CAPS
            if len(para_text) < 80 and para_text.isupper():
                is_heading = True
                heading_level = 1
            # Heuristic 2: Short line, Title Case, at top of page
            elif len(para_text) < 100 and i < 3 and para_text.istitle():
                is_heading = True
                heading_level = 2
            # Heuristic 3: Very short line, looks like section header
            elif len(para_text) < 50 and i < 5 and not para_text.endswith('.'):
                is_heading = True
                heading_level = 3
            # Heuristic 4: Check bounding box position (top of page, centered)
            elif para.layout.bounding_poly and len(para.layout.bounding_poly.normalized_vertices) >= 4:
                # Top of page (y close to 0) and roughly centered
                vertices = para.layout.bounding_poly.normalized_vertices
                avg_y = sum(v.y for v in vertices) / len(vertices)
                avg_x = sum(v.x for v in vertices) / len(vertices)
                if avg_y < 0.15 and 0.2 < avg_x < 0.8 and len(para_text) < 100:
                    is_heading = True
                    heading_level = 2
            
            if is_heading:
                prefix = "#" * min(heading_level, 6)
                markdown_parts.append(f"{prefix} {para_text}")
            else:
                # Regular paragraph
                markdown_parts.append(para_text)
        
        # Extract tables from this page
        for table in page.tables:
            table_md = _table_to_markdown(table, document.text)
            if table_md:
                markdown_parts.append(f"\n{table_md}\n")
    
    return "\n\n".join(markdown_parts)


def _get_text(layout, full_text: str) -> str:
    """Extract text from a Document AI layout object."""
    if not layout or not layout.text_anchor:
        return ""
    
    text_segments = []
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index) if segment.end_index else len(full_text)
        text_segments.append(full_text[start:end])
    
    return "".join(text_segments)


def _table_to_markdown(table, full_text: str) -> str:
    """
    Convert a Document AI table to markdown table format.
    """
    if not table.header_rows and not table.body_rows:
        return ""
    
    markdown_rows = []
    
    # Process header rows
    for row in table.header_rows:
        cells = []
        for cell in row.cells:
            cell_text = _get_text(cell.layout, full_text).strip().replace("\n", " ")
            cells.append(cell_text)
        markdown_rows.append("| " + " | ".join(cells) + " |")
        
        # Add separator row after first header row
        if len(markdown_rows) == 1:
            markdown_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    
    # Process body rows
    for row in table.body_rows:
        cells = []
        for cell in row.cells:
            cell_text = _get_text(cell.layout, full_text).strip().replace("\n", " ")
            cells.append(cell_text)
        markdown_rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(markdown_rows) if markdown_rows else ""


def list_doc_ai_processors():
    """
    List all Document AI processors in the project to help debug processor ID/location issues.
    """
    if not DOC_AI_AVAILABLE:
        raise ImportError("google-cloud-documentai not installed")
    
    load_dotenv()
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("DOC_AI_LOCATION", "australia-southeast1")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set")
    
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    
    parent = f"projects/{project_id}/locations/{location}"
    print(f"🔍 Listing processors in: {parent}")
    
    try:
        processors = client.list_processors(parent=parent)
        processors_list = list(processors)  # Convert pager to list
        print(f"\nFound {len(processors_list)} processor(s):")
        for p in processors_list:
            print(f"  - ID: {p.name.split('/')[-1]}")
            print(f"    Name: {p.display_name}")
            print(f"    Type: {p.type_}")
            print(f"    State: {p.state.name}")
            print(f"    Full path: {p.name}")
            print()
    except Exception as e:
        print(f"❌ Failed to list processors: {e}")
        raise

def summarize_doc(doc, llm):

    response = llm.invoke(f"""
            System: You are a helpful assistant that gives a topic to a document and describes it in a one or two sentences.\n
            
            User: Please write a topic for the following document, together with a few sentences description on what is it about:\n{doc}.

            Please keep the description concise and to the point (up to 50 words).

            Please only return the following:
            
            Topic: 

            Description:
            """)
    return str(response.content)


def parse_embedding(s):
    if isinstance(s, np.ndarray):
        return s
    if isinstance(s, list):
        return np.array(s, dtype=float)
    if not isinstance(s, str):
        raise ValueError(f"Unexpected embedding type: {type(s)}")

    # Remove brackets and extra spaces
    s = s.strip().replace("[", "").replace("]", "").replace(",", " ")
    # Split on spaces (one or more)
    nums = re.split(r"\s+", s.strip())
    # Convert to float safely
    result = []
    for x in nums:
        if x == "":
            continue
        try:
            result.append(float(x))
        except ValueError:
            raise ValueError(f"Invalid number '{x}' in embedding string: {s}")

    return np.array(result, dtype=float)

def find_relevant_document_fast(embedding_model, query, df, embed_col, top_k=3):
    """
    Quickly finds the top-k most relevant documents in a DataFrame based on cosine similarity 
    between the query embedding and document embeddings.

    Args:
        embedding_model: An embedding model with an `embed_query` method to generate query embeddings.
        query (str): The query string to search for relevant documents.
        df (pd.DataFrame): The DataFrame containing documents and their embeddings.
        embed_col (str): The name of the column in `df` containing document embeddings.
        top_k (int, optional): The number of top relevant documents to return. Default is 3.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k most relevant documents, sorted by similarity (descending).
    """
    query_emb = np.array(embedding_model.embed_query(query), dtype=float)
    df[embed_col] = df[embed_col].apply(parse_embedding)

    sims = df[embed_col].apply(lambda x: np.dot(x, query_emb) / (norm(x) * norm(query_emb)))
    top_docs = df.iloc[np.argsort(sims)[-top_k:][::-1]]
    return top_docs


def dict_to_doctext(dialogue_dict):
    lines = []
    for i in range(len(dialogue_dict['conv_questions'])):
        q = dialogue_dict['conv_questions'][i]
        reason = dialogue_dict['turn_program'][i]
        a = dialogue_dict['executed_answers'][i]
        lines.append(f"Q: {q}\nReason: {reason}\nA: {a}")
    return '"""\n' + '\n'.join(lines) + '\n"""'



def merge_metadata_into_content(docs, mode="prepend"):
    """
    Adds metadata info into each Document's page_content.
    
    Args:
        docs (list[Document]): List of LangChain Document objects.
        mode (str): "prepend" or "append" — whether to add metadata before or after the content.
    """
    new_docs = []

    for doc in docs:
        # Turn metadata dict into a readable string
        meta_text = " | ".join(f"{k}: {v}" for k, v in doc.metadata.items())

        if mode == "prepend":
            new_content = f"[{meta_text}]\n\n{doc.page_content}"
        else:
            new_content = f"{doc.page_content}\n\n[{meta_text}]"

        # Create a new Document with the updated text but keep metadata
        new_docs.append(Document(page_content=new_content, metadata=doc.metadata))
    
    return new_docs

def merge_metadata(meta1, meta2):
    """
    Merge two metadata dictionaries by appending values for duplicate keys.
    Returns a new dict.
    """
    merged = defaultdict(list)

    # Add first dict
    for k, v in meta1.items():
        merged[k].append(v)

    # Add second dict
    for k, v in meta2.items():
        merged[k].append(v)

    # Flatten lists if they only contain one element
    merged = {k: v[0] if len(v) == 1 else v for k, v in merged.items()}
    return merged


def merge_small_chunks(docs, min_length=300):
    """
    Merge consecutive small chunks in a list of LangChain Documents 
    until all chunks meet the minimum content length.
    
    Args:
        docs (list[Document]): List of Document objects (already chunked).
        min_length (int): Minimum acceptable length (in characters) for each chunk.
    
    Returns:
        list[Document]: List of merged Documents.
    """
    merged_docs = []
    buffer = None  # holds ongoing merge

    for doc in docs:
        # If there's nothing in buffer, start with this doc
        if buffer is None:
            buffer = doc
            continue

        # If buffer is too small, merge it with current doc
        if len(buffer.page_content) < min_length:
            combined_text = buffer.page_content + "\n\n" + doc.page_content
            combined_meta = merge_metadata(buffer.metadata, doc.metadata)
            buffer = Document(page_content=combined_text, metadata=combined_meta)
        else:
            merged_docs.append(buffer)
            buffer = doc

    # Add any leftover buffer
    if buffer:
        merged_docs.append(buffer)
    
    return merged_docs


def fin_doc_parser(document, embedding_model, min_chunk_length=500, docs=None, parser="docling"):
    """
    Parses a financial document into structured chunks, merges small chunks, and generates embeddings for each chunk.

    Args:
        document (str): Path to the financial document file to be parsed.
        embedding_model: An embedding model with an `embed_query` method for generating embeddings.
        min_chunk_length (int, optional): Minimum character length for each chunk after merging. Default is 500.
        docs (list[Document], optional): Pre-chunked list of LangChain Document objects. If None, the document is loaded and chunked.
        parser (str, optional): Parser to use - "docling" (local), "doc_ai" or "doc_ai_ocr" (Google Document AI OCR - free tier), "doc_ai_layout" (Layout Parser - paid), or "doc_ai_form" (legacy Form Parser - paid). Default is "docling".

    Returns:
        df_docs (pd.DataFrame): DataFrame with columns 'content', 'metadata', and 'content_embed' for each chunk.
        docs_with_meta (list[Document]): List of Document objects with metadata merged into content.
    """
    
    if docs is None:
        if parser == "doc_ai" or parser == "doc_ai_ocr":
            # Use Google Document AI Enterprise OCR (free tier: 1000 pages/month)
            try:
                pages = parse_with_doc_ai(document)
            except Exception as e:
                print(f"Google Document AI OCR failed: {e}. Falling back to docling...")
                parser = "docling"
        
        if parser == "docling":
            # 1. Force the accelerator device to CPU
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU,
                num_threads=2 # Optional: adjust threads based on your Mac core count
            )

            # 2. Assign these options to the PDF pipeline configuration
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options
            # Configure DocumentConverter properly using InputFormat enum keys
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            pages = converter.convert(document).document.export_to_markdown()

        # Step 1: Define your markdown header structure
        # You can include as many heading levels as needed
        headers_to_split_on = [
            ("#", "Section"),
            ("##", "Subsection"),
            ("###", "Subsubsection")
        ]

        # Step 2: Create the splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Step 3: Split the markdown text into structured chunks
        docs: list[Document] = markdown_splitter.split_text(pages)

        # Step 4: Assuming `docs` is from MarkdownHeaderTextSplitter
        docs_with_meta = merge_metadata_into_content(docs)
    else:
        docs_with_meta = docs

    merged_docs_with_meta = merge_small_chunks(docs_with_meta, min_length=min_chunk_length)

    df_docs = pd.DataFrame([
    {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in merged_docs_with_meta
    ])

    df_docs['content_embed'] = df_docs['content'].apply(lambda x: embedding_model.embed_query(x))

    return df_docs, docs_with_meta