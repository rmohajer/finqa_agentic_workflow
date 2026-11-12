
import asyncio
import os
from typing import (
    Annotated,
    Any,
    Literal,
)

# from langchain_mistralai import MistralAIEmbeddings
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

from src.parsing_helpers import (
    fin_doc_parser,
    find_relevant_document_fast,
    parse_embedding,
)
from src.prompt_helpers import system_message, user_message_new, user_message_tool
from src.tool_helpers import load_config, tool_names, tools, tools_by_name
from src.vector_db.weaviate_client import search_weaviate_db

"""
It is assumed that we use the same embedding model for both the ConvFinQA dataset and the user-uploaded financial report or any other RAG application used in this project. If you want to use different embedding models for the ConvFinQA dataset and the user-uploaded financial report, you can modify the code accordingly. You would need to create separate embedding model instances for each and use them in their respective contexts. For example, you could have one embedding model for the ConvFinQA dataset and another for the user-uploaded financial report. Just make sure to pass the correct embedding model to the relevant functions when processing each type of data.

Please use @data_parsing_experiment.ipynb to parse the ConvFinQA dataset and generate embeddings. The notebook will create a new CSV file with the parsed data and embeddings, which can then be used in this app.py code.

"""

config = load_config()

load_dotenv()

# Shared Weaviate client for FinQA example retrieval (created once, reused by call_model_node)
_weaviate_client = None

def create_weaviate_client():
    """Create the shared Weaviate Cloud client once; returns None if creds are missing."""
    global _weaviate_client
    if _weaviate_client is not None:
        return _weaviate_client
    weaviate_url = config.get("WEAVIATE_URL") or os.getenv("WEAVIATE_URL")
    weaviate_api_key = config.get("WEAVIATE_API_KEY") or os.getenv("WEAVIATE_API_KEY")
    if not (weaviate_url and weaviate_api_key):
        print("⚠️  WEAVIATE_URL / WEAVIATE_API_KEY not set — using in-memory search")
        return None
    try:
        import weaviate
        from weaviate.classes.init import Auth
        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            skip_init_checks=False,
        )
        print("✅ Weaviate client connected for FinQA example retrieval")
    except Exception as e:
        print(f"⚠️  Weaviate connection failed: {e} — using in-memory search")
        return None
    return _weaviate_client

# Load the FinQA dataset

df_finqa = pd.read_csv("data/convfinqa_wembed.csv")

# Parse the 'summary_emb' column from JSON strings to numpy arrays (old embeddings, will be replaced)
df_finqa["summary_emb"] = df_finqa["summary_emb"].apply(parse_embedding)


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

llm=llm_groq  # Default LLM for the app, can be switched to llm_nvidia if needed

# Default: Nemotron 3.5 Embed 1B via NVIDIA API (developer-configured)
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Create embedding model
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
embedding_model = NVIDIAEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    api_key=os.getenv("NVIDIA_EMBEDDING_API_KEY"),
    truncate="END",
)

# # Cache file for ConvFinQA embeddings (avoids recomputing on every restart)
# EMBEDDINGS_CACHE_PATH = Path(__file__).parent.parent / "data" / "convfinqa_embeddings_cache.npy"

# def _load_or_compute_finqa_embeddings():
#     """
#     Load ConvFinQA embeddings from cache, or compute and cache them lazily.
#     Uses batch embedding API for efficiency.
#     """
#     global df_finqa
    
#     # Check if cache exists and is valid
#     if EMBEDDINGS_CACHE_PATH.exists():
#         try:
#             print("📦 Loading ConvFinQA embeddings from cache...")
#             cached_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
#             if len(cached_embeddings) == len(df_finqa):
#                 df_finqa["summary_emb"] = list(cached_embeddings)
#                 print(f"✅ Loaded {len(df_finqa)} embeddings from cache (dim: {len(cached_embeddings[0])})")
#                 return
#             else:
#                 print(f"⚠️  Cache size mismatch ({len(cached_embeddings)} vs {len(df_finqa)}), recomputing...")
#         except Exception as e:
#             print(f"⚠️  Failed to load cache: {e}, recomputing...")
    
#     # Compute embeddings using batch API (much faster than sequential)
#     print("🔄 Computing ConvFinQA embeddings with Nemotron model (first run, this takes ~2-3 minutes)...")
#     texts = df_finqa["full_example"].tolist()
    
#     try:
#         # Use batch embedding for efficiency
#         embeddings = embedding_model.embed_documents(texts)
#         embeddings_array = np.array(embeddings)
        
#         # Save to cache
#         np.save(EMBEDDINGS_CACHE_PATH, embeddings_array)
#         print(f"💾 Saved embeddings cache to {EMBEDDINGS_CACHE_PATH}")
#     except Exception as e:
#         print(f"⚠️  Batch embedding failed ({e}), falling back to sequential...")
#         embeddings = []
#         for i, text in enumerate(texts):
#             if i % 100 == 0:
#                 print(f"  Progress: {i}/{len(texts)}")
#             embeddings.append(embedding_model.embed_query(text))
#         embeddings_array = np.array(embeddings)
#         np.save(EMBEDDINGS_CACHE_PATH, embeddings_array)
    
#     df_finqa["summary_emb"] = list(embeddings_array)
#     print(f"✅ Computed {len(df_finqa)} ConvFinQA embeddings (dim: {len(embeddings_array[0])})")

# # Initialize embeddings lazily on first access
# # Use a flag to track initialization
# _finqa_embeddings_initialized = False

# def _ensure_finqa_embeddings():
#     """Ensure ConvFinQA embeddings are initialized (lazy initialization)."""
#     global _finqa_embeddings_initialized
#     if not _finqa_embeddings_initialized:
#         _load_or_compute_finqa_embeddings()
#         _finqa_embeddings_initialized = True

# Don't initialize embeddings at import time - do it lazily when first needed
# _ensure_finqa_embeddings()  # Commented out for fast startup

min_chunk_length = MIN_CHUNK_LENGTH


class FinState(MessagesState):
    """ The state of the AI Assistant"""
    report_state: Annotated[bool, "Whether the financial report is provided by the user"]
    is_finqa_doc: Annotated[bool, "Whether the uploaded document is a financial/FinQA-related document"]
    user_query: Annotated[list[str], "The list of questions from the user in continuous chat"]
    next_node: Annotated[str | None, "The next node to execute, either 'tool_node' or 'call_model_node'"]
    call_mode: Annotated[Literal["user","tool_call"], "The current mode of the call, either 'user' for model calls or 'tool_call' for tool executions"]
    context_content: Annotated[str, "The relevant content from the financial report to assist in answering the user's query"]
    context_examples: Annotated[str | None, "The relevant example conversations from the FinQA dataset to assist in answering the user's query"]
    message_out: Annotated[str | None, "The final answer to the user's query, if available"]
    full_response: Annotated[Any | None, "The full response from the model, including any tool calls or reasoning steps"]
    query_cont: Annotated[list[str], "Whether this is a new query or a continuation of the previous ones"]

def check_info_node(state: FinState):
    """Check if the input document context is provided by the user"""
    # Simple heuristic: if context_docs is empty or too short, we consider it insufficient
    if not state["report_state"]:

        state["next_node"] = "END"
        state["message_out"] = "I'm sorry, but I don't have enough information to answer your question. Please provide the necessary financial documents."
    else:
        state["next_node"] = "call_model_node"

    # chekc if this is a new query or continuation
    prompt_template = PromptTemplate.from_template("This is the new query by the user: {new_query}. This is the previous queries by the user: {prev_queries}. Is the new query a continuation of the previous queries or a new one? Only Answer Yes or No.")

    new_query = state["user_query"][-1]
    prev_queries = "\n".join(state["user_query"][:-1])

    chain = prompt_template|llm
    response = chain.invoke({"new_query": new_query,
                                "prev_queries": "\n".join(prev_queries)})

    print("A continuation Question? ", response.content)
    if "yes" in response.content.lower():
        state["query_cont"].append(new_query)
    else:
        state["query_cont"] = [new_query]

    print("connected queries: \n ", state["query_cont"])
    return {**state,
            "message_out": state["message_out"],
            "next_node": state["next_node"],
            "query_cont": state["query_cont"],
            "messages": []}


def classify_doc_node(state: FinState):
    """Classify if the uploaded document is a financial/FinQA-related document."""
    if not state["report_state"]:
        state["is_finqa_doc"] = False
        state["next_node"] = "call_model_node"
        return {**state, "is_finqa_doc": state["is_finqa_doc"], "next_node": state["next_node"]}
    
    # Get df_docs from global (set by FinQAResponseAgent.predict)
    global df_docs
    if df_docs is None or df_docs.empty:
        state["is_finqa_doc"] = False
        state["next_node"] = "call_model_node"
        return {**state, "is_finqa_doc": state["is_finqa_doc"], "next_node": state["next_node"]}
    
    # Sample first few chunks for classification
    sample_chunks = df_docs["content"].head(5).tolist()
    sample_text = "\n\n---\n\n".join(sample_chunks)
    
    prompt_template = PromptTemplate.from_template(
        "Analyze the following document excerpts. Is this a financial report, earnings statement, "
        "SEC filing, financial statement, or similar financial document? Answer only 'Yes' or 'No'.\n\n"
        "Document excerpts:\n{sample_text}"
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"sample_text": sample_text})
    
    is_financial = "yes" in response.content.lower()
    state["is_finqa_doc"] = is_financial
    state["next_node"] = "call_model_node"
    
    print(f"Document classified as financial/FinQA: {is_financial}")
    
    return {**state, "is_finqa_doc": state["is_finqa_doc"], "next_node": state["next_node"]}

def should_continue_chat(state: FinState):
    """Determine whether to continue the chat or end the cycle."""
    # If there is no function call, then we finish
    if state["next_node"]=="END":
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def tool_node(state: FinState):
    """Execute all tool calls from the last message in the state."""
    outputs = []
    for tool_call in state["full_response"].tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name[tool_name]
        tool_call_id = tool_call["id"]
        tool_args = tool_call["args"]
        tool_response = tool.invoke(tool_args)
        tool_message= f"Tool {tool_name} invoked with args {tool_args}, got response: {tool_response} \n"
        print(tool_message)
        outputs.append(
            ToolMessage(
                content=tool_message,
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        )
    return {"messages": outputs}


def call_model_node(state: FinState):
    """Invoke the model with the current conversation state."""
    if state["call_mode"]=="user":
        query = state["user_query"][-1]
        queries = "\n".join(state["query_cont"])
        relevant_docs = find_relevant_document_fast(embedding_model, queries, df_docs, "content_embed", top_k=4)
        
        context_content = "\n".join([doc for doc in relevant_docs["content"]])
        state["context_content"] = context_content
        # Only retrieve FinQA examples if document is financial/FinQA-related
        if state.get("is_finqa_doc", False):
            weaviateclient = create_weaviate_client()
            context_examples = None
            if weaviateclient is not None:
                try:
                    collection_name = config.get("WEAVIATE_COLLECTION_NAME", "ExampleFinQA")
                    # Fields to fetch come from config (per-collection), so this
                    # call stays generic for any vectordb defined in Weaviate
                    return_fields = (config.get("WEAVIATE_COLLECTION_FIELDS", {}) or {}).get(collection_name)
                    if isinstance(return_fields, str):
                        return_fields = [f.strip() for f in return_fields.split(",") if f.strip()]
                    results = search_weaviate_db(
                        client=weaviateclient,
                        collection_name=collection_name,
                        query=queries,
                        top_k=2,
                        search_mode="hybrid",
                        alpha=0.75,
                        return_properties=return_fields,
                        close_client=False,
                    )
                    text_field = config.get("WEAVIATE_TEXT_FIELD", "full_example")
                    context_examples = "\n".join([
                        f"Example {i}: {r.get(text_field, '')}" 
                        for i, r in enumerate(results)
                    ])
                except Exception as e:
                    print(f"⚠️  Weaviate search failed: {e}, falling back to in-memory search")

            if context_examples is None:
                # Fallback to in-memory search
                relevant_finq_docs = find_relevant_document_fast(embedding_model, queries, df_finqa, "summary_emb", top_k=2)
                context_examples = "\n".join([f"Example {i}: {doc}" for i,doc in enumerate(relevant_finq_docs["full_example"])]) 
            state["context_examples"] = context_examples
        else:
            state["context_examples"] = ""
        
        user_message = user_message_new
    else:
        user_message = user_message_tool
        query = state["user_query"][-1]
        context_content = state["context_content"]
        context_examples = state["context_examples"]

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        user_message
    ])

    model_react=chat_prompt|llm.bind_tools(tools)

    response = model_react.invoke({
        "context_content": context_content
        , "context_examples": state["context_examples"] 
        , "chat_history": state["messages"]
        , "tool_names": ", ".join(tool_names)
        , "query": query
        })

    state["full_response"] = response
    if not response.tool_calls:
        state["call_mode"] = "user"
        state["message_out"] = response.content
        out_message = [response.content]
    else:
        state["call_mode"] = "tool_call"
        print(response.tool_calls)
        out_message = []



    return {**state,
            "call_mode": state["call_mode"],
            "context_content": state["context_content"],
            "context_examples": state["context_examples"],
            "message_out": state["message_out"],
            "full_response": state["full_response"],
            "messages": out_message}


def should_continue_react(state: FinState):
    """Determine whether to continue with tool use or end the ReAct loop."""
    call_mode = state["call_mode"]
    # If there is no function call, then we finish
    if call_mode=="user":
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


builder = StateGraph(FinState)
builder.add_node("check_info_node", check_info_node)
builder.add_node("classify_doc_node", classify_doc_node)
builder.add_node("call_model_node", call_model_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "check_info_node")
builder.add_conditional_edges("check_info_node",
                              should_continue_chat, 
                              {"continue": "classify_doc_node",
                               "end": END})
builder.add_edge("classify_doc_node", "call_model_node")
builder.add_conditional_edges("call_model_node", 
                              should_continue_react, 
                              {"continue": "tool_node",
                               "end": END})
builder.add_edge("tool_node", "call_model_node")

finqa_graph = builder.compile(checkpointer=InMemorySaver())



class FinQAResponseAgent:
    """Async wrapper for running the finqa_graph ReAct agent as a reusable service."""

    def __init__(self, agent_graph, report_state: bool = False, config: dict | None = None):
        """
        Initialize the response agent.

        Args:
            agent_graph: Compiled LangGraph instance (e.g., finqa_graph)
            report_state: Whether the financial report is provided by the user
            config: Configuration dictionary (optional, loads from config file if not provided)
        """
        self.agent = agent_graph
        self.report_state = report_state
        self.df_docs = None  # To be set after PDF upload
        self.session_store = {}  # In-memory storage for session data
        
        # Create the shared Weaviate client up front so call_model_node can use it
        self.config = config or load_config()
        self.weaviate_client = create_weaviate_client()

    def set_report_state(self, value: bool):
        """Update report_state dynamically (e.g., after PDF upload)."""
        self.report_state = value

    def set_documents(self, df_docs):
        """Attach parsed financial report (and optionally embedding model) to the agent."""
        self.df_docs = df_docs

    def _get_session_data(self, session_id: str) -> dict[str, Any]:
        """Get session data for a given session ID."""
        return self.session_store.get(session_id, {"df_docs": None, "report_state": False})

    def _set_session_data(self, session_id: str, data: dict[str, Any]):
        """Set session data for a given session ID."""
        self.session_store[session_id] = data

    async def handle_pdf_upload(self, file_obj, session_id: str, min_chunk_length: int, embedding_model, parser: str = "docling") -> str:
        """
        Asynchronously parse PDF and store text embeddings.
        
        Args:
            file_obj: The uploaded PDF file object
            session_id: Unique session identifier
            min_chunk_length: Minimum chunk length for merging small chunks
            embedding_model: Embedding model to use for generating embeddings
            parser: Parser to use ('docling' or 'doc_ai')
            
        Returns:
            Status message string
        """
        if file_obj is None:
            self.set_report_state(False)
            self._set_session_data(session_id, {"df_docs": None, "report_state": False})
            return "❌ Please upload a PDF file first."

        try:
            df_docs, docs = await asyncio.to_thread(
                fin_doc_parser, file_obj, embedding_model, min_chunk_length, None, parser
            )

            if df_docs is None or df_docs.empty:
                self.set_report_state(False)
                self._set_session_data(session_id, {"df_docs": None, "report_state": False})
                return "⚠️ No readable text found in the PDF."
            else:
                self.set_report_state(True)
                self.set_documents(df_docs)
                self._set_session_data(session_id, {"df_docs": df_docs, "report_state": True})

            return "✅ PDF parsed successfully!"
        except Exception as e:
            self.set_report_state(False)
            self._set_session_data(session_id, {"df_docs": None, "report_state": False})
            return f"❌ Error during PDF parsing: {str(e)}"

    def convert_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert user input to a format acceptable by finqa_graph.

        Args:
            input_data (dict): {'user_id': ..., 'user_input': ...}

        Returns:
            dict: Input state for the LangGraph.
        """
        user_input = input_data.get("user_input", "")
        user_id = input_data.get("user_id", "default_user")

        # Build run config for this user
        run_config = {"configurable": {"thread_id": user_id}}

        # Check if a state already exists (for continuity)
        existing_state = self.agent.get_state(config=run_config)

        # Build initial state only if needed
        if existing_state is None or not existing_state.values:
            state = {
                "messages": [],
                "report_state": self.report_state,  
                "is_finqa_doc": False,  # Will be determined by classify_doc_node
                "user_query": [user_input],
                "next_node": None,
                "call_mode": "user",
                "context_content": "",
                "context_examples": "",
                "message_out": None,
                "full_response": None,
            }
            state["messages"] = add_messages(state["messages"], HumanMessage(content=user_input))
        else:
            state = existing_state.values
            state["user_query"].append(user_input)
            state["report_state"] = self.report_state  # Ensure it's updated
            # Preserve is_finqa_doc from existing state


        state["messages"] = add_messages(state["messages"], HumanMessage(content=user_input))

        return {"state": state, "run_config": run_config}

    async def predict(self, input_data: dict[str, Any]) -> str:
        """
        Run the finqa_graph and return the model output.

        Args:
            input_data (dict): {'user_id': ..., 'user_input': ...}

        Returns:
            str: The model's response (message_out)
        """
        if self.df_docs is not None:
            globals()["df_docs"] = self.df_docs

        converted = self.convert_input(input_data)
        state = converted["state"]
        run_config = converted["run_config"]

        # Run the graph asynchronously
        agent_response = await self.agent.ainvoke(state, config=run_config)

        # Retrieve the model output
        message_out = agent_response["message_out"]

        return {"output": message_out,
                "full_response": agent_response["full_response"]
        }


FinQAAgent = FinQAResponseAgent(finqa_graph, config=config)
