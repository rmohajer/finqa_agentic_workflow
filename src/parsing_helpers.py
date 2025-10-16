import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# from langchain_mistralai import MistralAIEmbeddings
import pandas as pd
import json
from langchain.embeddings import HuggingFaceEmbeddings
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
from langchain.schema import Document
from collections import defaultdict
from docling.document_converter import DocumentConverter

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


def fin_doc_parser(document, embedding_model, min_chunk_length=500, docs=None):
    """
    Parses a financial document into structured chunks, merges small chunks, and generates embeddings for each chunk.

    Args:
        document (str): Path to the financial document file to be parsed.
        embedding_model: An embedding model with an `embed_query` method for generating embeddings.
        min_chunk_length (int, optional): Minimum character length for each chunk after merging. Default is 500.
        docs (list[Document], optional): Pre-chunked list of LangChain Document objects. If None, the document is loaded and chunked.

    Returns:
        df_docs (pd.DataFrame): DataFrame with columns 'content', 'metadata', and 'content_embed' for each chunk.
        docs_with_meta (list[Document]): List of Document objects with metadata merged into content.
    """
    
    if docs is None:
        converter = DocumentConverter()
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