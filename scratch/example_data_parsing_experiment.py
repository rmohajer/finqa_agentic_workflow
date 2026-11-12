# Generated from: example_data_parsing_experiment.ipynb
# Converted at: 2026-11-12T00:14:44.503Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

#%% # Notebook for data analysis and Cleaning


import os
import yaml
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# from langchain_mistralai import MistralAIEmbeddings
import pandas as pd

import numpy as np
import sys
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load config from YAML file (works in notebook context)
def load_config():
    # Get the notebook directory
    notebook_dir = Path(os.getcwd())
    config_path = notebook_dir.parent / "config" / "config-common-yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Configuration from config file
NVIDIA_MODEL = config["NVIDIA_MODEL"]
GROQ_MODEL = config["GROQ_MODEL"]
LLM_TEMPERATURE = config["LLM_TEMPERATURE"]
LLM_MAX_TOKENS = config["LLM_MAX_TOKENS"]
EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
NVIDIA_CHAT_TEMPLATE_KWARGS = config["NVIDIA_CHAT_TEMPLATE_KWARGS"]
MIN_CHUNK_LENGTH = config["MIN_CHUNK_LENGTH"]


nvidia_api_key = os.getenv("NVIDIA_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

client = ChatNVIDIA(
        model=NVIDIA_MODEL,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        seed=42,
    )

llm_nvidia = client.bind(**{"chat_template_kwargs": {"enable_thinking": False}, "parallel_tool_calls": True})

llm_groq = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# Default: Nemotron 3.5 Embed 1B via NVIDIA API (developer-configured)
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# Create embedding model
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
embedding_model = NVIDIAEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    api_key=os.getenv("NVIDIA_EMBEDDING_API_KEY"),
    truncate="END",
)


# Weaviate credentials from environment
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_COLLECTION_NAME = config.get("WEAVIATE_COLLECTION_NAME", "ExampleFinQA")
WEAVIATE_BATCH_SIZE = config.get("WEAVIATE_BATCH_SIZE", 100)

print(f"Weaviate URL: {WEAVIATE_URL}")
print(f"Collection: {WEAVIATE_COLLECTION_NAME}")
print(f"Batch size: {WEAVIATE_BATCH_SIZE}")


#%% ## Initial data processing


import pandas as pd
import json

# Read the JSON file
with open('../data/convfinqa_dataset.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df_train = pd.DataFrame(data['train'])
def_dev = pd.DataFrame(data['dev'])
df = pd.concat([df_train, def_dev], ignore_index=True)

# Display the first few rows
print('Shape of the dataframe: ', df.shape)
df.head()

#%%

import json

def _parse(x):
    return json.loads(x) if isinstance(x, str) else x

df["full_example"] = df.apply(
    lambda r: json.dumps(
        {
            "doc": _parse(r["doc"]),
            "dialogue": _parse(r["dialogue"]),
            "features": _parse(r["features"]),
        },
        ensure_ascii=False,
    ),
    axis=1,
)

df.iloc[20]['full_example']

#%% ## Summarize each document 


from joblib import Parallel, delayed
import math
import time



def summarize_doc(doc,llm):

    response = llm.invoke(f"""
            System: You are a helpful assistant that writes a topic for a financial question scenario, together with a summary of the problem and key concepts covered by the scenario.
            This extracted information helps in building a knowledge base for financial question answering, in a way that the approrpriate examples will be extracted from the knowledge base based on the similarity of the query to the summary that you are writing on this exmaple.
            \n
            
            User: Financial question scenario:\n
            
            {doc}.

            Please keep the description concise and to the point (up to 200 words).

            Please return the following:
            
            Topic: 

            Summary Description:

            Key concepts covered by this example:
            """)
    return str(response.content)



def summarize_doc_with_retry(doc, llm, max_retries=3, retry_delay=5):
    attempt = 0
    while True:
        try:
            return summarize_doc(doc, llm)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            print(f"  Error: {e}. Retrying in {retry_delay}s (attempt {attempt}/{max_retries})...")
            time.sleep(retry_delay)

# for parallel processing need to use another technology like joblib 

doc_samples = df.iloc[200]['full_example']

summary_groq = summarize_doc(doc_samples,llm_groq)
summary_nvidia = summarize_doc(doc_samples,llm_nvidia)

print(f"Summary from Groq LLM:\n{summary_groq}\n")
print(f"Summary from Nvidia LLM:\n{summary_nvidia}\n")

batch_size = 20  # You can adjust this value
num_batches = math.ceil(len(df) / batch_size)

summaries = []
for i in range(num_batches):
    batch = df.iloc[i*batch_size : (i+1)*batch_size]
    batch_summaries = Parallel(n_jobs=-1, backend="threading")(delayed(summarize_doc_with_retry)(doc, llm_nvidia) for doc in batch['full_example'])
    summaries.extend(batch_summaries)
    print(f'Batch {i+1} of {num_batches} done.')
    time.sleep(5)  # To avoid hitting rate limits


df['summary'] = summaries

print('Shape of the dataframe: ', df.shape)
print(f'Length of summaries: {len(summaries)}')

df.to_csv('../data/convfinqa_wsummary.csv', index=False)

#%% ## Add embeddings


# Load the DataFrame from the CSV file
df_s = pd.read_csv('../data/convfinqa_wsummary.csv')
df_s.head()

# create embedding for the summaries with retry mechanism
import time

def embed_with_retry(text, max_retries=3, delay=5):
    """Embed text with retry on 502 Bad Gateway errors."""
    for attempt in range(max_retries):
        try:
            return embedding_model.embed_query(text)
        except Exception as e:
            if "502" in str(e) or "Bad Gateway" in str(e):
                if attempt < max_retries - 1:
                    print(f"  ⚠️  502 Bad Gateway on attempt {attempt + 1}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            raise


batch_size = 20  # You can adjust this value
num_batches = math.ceil(len(df_s) / batch_size)

embeddings = []
for i in range(num_batches):
    batch = df.iloc[i*batch_size : (i+1)*batch_size]
    batch_embeddings = Parallel(n_jobs=-1, backend="threading")(delayed(embed_with_retry)(text) for text in batch['summary'])
    embeddings.extend(batch_embeddings)
    print(f'Batch {i+1} of {num_batches} done.')
    

df_s["summary_emb"] = embeddings

# df_s["summary_emb"] = df_s["summary"].apply(lambda x: embed_with_retry(x))



#%% simple search through the data
row = df_s[df_s['features'].apply(lambda x: eval(str(x))['has_type2_question'])].iloc[30]

print('summary: ', row['summary'])
print('Features: ', row['features'])
print('Dialogue: ', row['dialogue'])

df_s.to_csv('../data/convfinqa_wembed.csv', index=False)
df_s.head()

#%% # ## Search relevant summaries as a function 

from src.parsing_helpers import parse_embedding
# Load the DataFrame from the CSV file
df_f = pd.read_csv('../data/convfinqa_wembed.csv')

print(type(df_f["summary_emb"].iloc[20]))
print(df_f["summary_emb"].iloc[20][0:60], '...') 
print('Need to parse the embeddings from string to list of floats')

import re
import numpy as np



df_f["summary_emb"] = df_f["summary_emb"].apply(parse_embedding)


print(type(df_f["summary_emb"].iloc[20]))
print(df_f["summary_emb"].iloc[20][0:60], '...') 


#%% 

import numpy as np
from numpy.linalg import norm

from src.parsing_helpers import find_relevant_document_fast


query = "What was S&P 500's stock price trend in the last quarter in 2012?"
relevant_docs = find_relevant_document_fast(embedding_model, query, df_f, 'summary_emb', top_k=3)
for i, row in relevant_docs.iterrows():
    print(f"Document {i}:")
    print("Summary:", row['summary'])
    print("Dialogue:", row['dialogue'])
    print("Features:", row['features'])
    print("\n---\n")    


#%% ## Restructure the FinQA dataset


import ast

# Extract the dictionary from the string in df_e.iloc[0]['dialogue']
features_str = df_f.iloc[0]['features']
features_dict = ast.literal_eval(features_str)
features_dict_keys = ast.literal_eval(df_f.iloc[0]['features']).keys()
print(features_dict_keys)
print(type(features_dict))  # Should be <class 'dict'>
print(features_dict)



# For each key in the first dictionary, create a new column
for key in features_dict_keys:
    df_f[key] = df_f['features'].apply(lambda x: ast.literal_eval(x)).apply(lambda d: d[key])

df_f.head()

df_f.to_csv('../data/convfinqa_full.csv', index=False)

########################################
#%% ## Weaviate Vector Database Ingestion
# 
# Ingest the ConvFinQA examples using Weaviate's built-in Snowflake Arctic Embed vectorizer.



#%% Initialize Weaviate Client
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in environment")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=False
)

print(f"Connected: {client.is_ready()}")

#%% ## Create Collection Schema
collections = client.collections.list_all()
if WEAVIATE_COLLECTION_NAME in collections:
    print(f"Collection {WEAVIATE_COLLECTION_NAME} already exists, deleting...")
    client.collections.delete(WEAVIATE_COLLECTION_NAME)

client.collections.create(
    name=WEAVIATE_COLLECTION_NAME,
    description="ConvFinQA financial question-answering examples with Weaviate Snowflake Arctic Embed",
    properties=[
        Property(name="record_id", data_type=DataType.TEXT, description="Original record ID", index_filterable=True, index_searchable=True),
        Property(name="summary", data_type=DataType.TEXT, description="LLM-generated summary", index_filterable=True, index_searchable=True),
        Property(name="conv_questions", data_type=DataType.TEXT_ARRAY, description="Conversation questions", index_filterable=True, index_searchable=True),
        Property(name="conv_answers", data_type=DataType.TEXT_ARRAY, description="Conversation answers", index_filterable=False, index_searchable=True),
        Property(name="turn_program", data_type=DataType.TEXT_ARRAY, description="Program/execution steps", index_filterable=False, index_searchable=True),
        Property(name="executed_answers", data_type=DataType.NUMBER_ARRAY, description="Executed numeric answers", index_filterable=False, index_searchable=False),
        Property(name="qa_split", data_type=DataType.BOOL_ARRAY, description="Question split type flags", index_filterable=True, index_searchable=False),
        Property(name="num_dialogue_turns", data_type=DataType.INT, description="Number of dialogue turns", index_filterable=True, index_searchable=False),
        Property(name="has_type2_question", data_type=DataType.BOOL, description="Has type-2 questions", index_filterable=True, index_searchable=False),
        Property(name="has_duplicate_columns", data_type=DataType.BOOL, description="Has duplicate columns", index_filterable=True, index_searchable=False),
        Property(name="has_non_numeric_values", data_type=DataType.BOOL, description="Has non-numeric values", index_filterable=True, index_searchable=False),
        Property(name="full_example", data_type=DataType.TEXT, description="Full JSON of doc + dialogue + features", index_filterable=False, index_searchable=True),
    ],
    # Use Weaviate's built-in Snowflake Arctic Embed vectorizer
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
    # Enable hybrid search (BM25 + vector)
    inverted_index_config=Configure.inverted_index(index_null_state=True, index_property_length=True, index_timestamps=True)
)

collection = client.collections.get(WEAVIATE_COLLECTION_NAME)
print(f"Collection {WEAVIATE_COLLECTION_NAME} created successfully")

#%%  ## Ingest Data (Vectors generated automatically by Weaviate)
import pandas as pd
import ast
import numpy as np
from src.parsing_helpers import parse_array, parse_bool_array

# Load the full dataset
df_full = pd.read_csv('../data/convfinqa_full.csv')
print(f"Loaded {len(df_full)} records from convfinqa_full.csv")



#%% Prepare objects for batch insertion (no vectors needed - generated by Weaviate)
objects_to_insert = []
for _, row in df_full.iterrows():
    properties = {
        "record_id": str(row.get("id", "")),
        "summary": str(row.get("summary", "")),
        "conv_questions": parse_array(row.get("conv_questions")),
        "conv_answers": parse_array(row.get("conv_answers")),
        "turn_program": parse_array(row.get("turn_program")),
        "executed_answers": parse_array(row.get("executed_answers"), dtype=float),
        "qa_split": parse_bool_array(row.get("qa_split")),
        "num_dialogue_turns": int(row.get("num_dialogue_turns", 0)) if pd.notna(row.get("num_dialogue_turns")) else 0,
        "has_type2_question": bool(row.get("has_type2_question", False)) if pd.notna(row.get("has_type2_question")) else False,
        "has_duplicate_columns": bool(row.get("has_duplicate_columns", False)) if pd.notna(row.get("has_duplicate_columns")) else False,
        "has_non_numeric_values": bool(row.get("has_non_numeric_values", False)) if pd.notna(row.get("has_non_numeric_values")) else False,
        "full_example": str(row.get("full_example", "")),
    }
    
    objects_to_insert.append({
        "properties": properties
        # No vector - generated automatically by Weaviate
    })

print(f"Prepared {len(objects_to_insert)} objects for insertion")

########################################
#%% Batch Insert with Progress
batch_size = WEAVIATE_BATCH_SIZE
total = len(objects_to_insert)
inserted = 0
failed = 0

for i in range(0, total, batch_size):
    batch = objects_to_insert[i:i + batch_size]
    
    try:
        with collection.batch.dynamic() as batch_writer:
            for obj in batch:
                batch_writer.add_object(
                    properties=obj["properties"]
                    # No vector - generated automatically by Weaviate
                )
        inserted += len(batch)
    except Exception as e:
        print(f"Batch {i//batch_size + 1} failed: {e}")
        failed += len(batch)
    
    if (i // batch_size + 1) % 10 == 0 or i + batch_size >= total:
        print(f"Progress: {min(i + batch_size, total)}/{total} (inserted: {inserted}, failed: {failed})")

print(f"\nIngestion complete: {inserted} inserted, {failed} failed")

#%% Verify Ingestion with Test Queries (Hybrid search works automatically)
test_queries = [
    "cash flow from operating activities",
    "revenue growth year over year",
    "share repurchase program",
    "gross margin trend",
    "net sales change percentage"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = collection.query.hybrid(
        query=query,
        alpha=0.75,
        limit=3,
        return_properties=["record_id", "summary", "num_dialogue_turns", "has_type2_question"]
    )
    for i, obj in enumerate(results.objects):
        print(f"  {i+1}. [{obj.properties.get('record_id', 'N/A')}] {obj.properties.get('summary', '')[:100]}... (score: {obj.metadata.score})")

# Close Connection
client.close()
print("Weaviate connection closed")


#%%
# Example usage:
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from src.vector_db.weaviate_client import search_weaviate_db

client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
    )
collection_name = WEAVIATE_COLLECTION_NAME
query = "cash flow from operations"

results = search_weaviate_db(
    client, collection_name, query,
    top_k=3,
    search_mode="hybrid",
    return_properties=["record_id", "summary", "full_example", "num_dialogue_turns", "has_type2_question"],
)
for r in results:
    print(f"{r['record_id']}: {r['summary'][:80]}... (score: {r['score']})")
client.close()


# %%
