"""
Weaviate Cloud Vector Database Client for ConvFinQA Examples.

Provides a Python wrapper for connecting to Weaviate Cloud, managing collections,
ingesting data with Weaviate's built-in Snowflake Arctic Embed vectorizer,
and performing vector/hybrid search operations.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from Weaviate."""
    uuid: str
    summary: str
    score: float
    metadata: dict[str, Any]




def search_weaviate_db(client, collection_name, query: str, top_k: int = 3, filters: dict = None, search_mode: str = "hybrid", alpha: float = 0.75, return_properties: list = None, close_client: bool = False):
    """Search a Weaviate collection using hybrid or vector search (collection-agnostic).
    
    Which properties are fetched is decided by the caller via return_properties,
    so the same function works for any collection schema in the project.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection to search
        query: Search query text
        top_k: Number of results to return
        filters: Optional metadata filters
        search_mode: Search mode - "hybrid" (BM25 + vector) or "vector" (pure vector).
            Note: hybrid search is only run when filters are provided; otherwise
            the query falls back to a pure vector (near_text) search.
        alpha: Hybrid search weight (0=BM25, 1=vector), default 0.75
        return_properties: List of property names to fetch and return.
            None -> all collection properties are returned.
        close_client: Whether to close the client connection after searching.
            Keep False when reusing a shared/persistent client.
    """
    try:
        collection = client.collections.get(collection_name)
        
        where_filter = None
        if filters:
            from weaviate.classes.query import Filter
            filter_conditions = []
            for key, value in filters.items():
                if isinstance(value, bool):
                    filter_conditions.append(Filter.by_property(key).equal(value))
                elif isinstance(value, (int, float)):
                    filter_conditions.append(Filter.by_property(key).equal(value))
                elif isinstance(value, str):
                    filter_conditions.append(Filter.by_property(key).equal(value))
            if filter_conditions:
                where_filter = filter_conditions[0]
                for fc in filter_conditions[1:]:
                    where_filter = where_filter & fc
        
        # Perform search based on mode
        if (where_filter and search_mode == "hybrid"):
            results = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=top_k,
                filters=where_filter,
                return_properties=return_properties
            )
        else:
            results = collection.query.near_text(
                query=query,
                limit=top_k,
                filters=where_filter,
                return_properties=return_properties
            )
        
        return [
            {
                **({p: obj.properties.get(p) for p in return_properties} if return_properties else obj.properties),
                "score": obj.metadata.score,
            }
            for obj in results.objects
        ]
    finally:
        if close_client:
            client.close()
