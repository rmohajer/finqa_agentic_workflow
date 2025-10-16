
import asyncio
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# from langchain_mistralai import MistralAIEmbeddings
import pandas as pd
import json
import numpy as np
import sys
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from typing import  (List, Annotated,Sequence,TypedDict,Any,Optional,Dict,Union,Literal)
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt, Command
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from parsing_helpers import find_relevant_document_fast, parse_embedding
from tool_helpers import tools, tools_by_name, tool_names
from prompt_helpers import system_message, user_message_new, user_message_tool


load_dotenv()
# Load the FinQA dataset

df_finqa_0 = pd.read_csv('../data/convfinqa_full.csv')

# Parse the 'summary_emb' column from JSON strings to numpy arrays
df_finqa_0["summary_emb"] = df_finqa_0["summary_emb"].apply(parse_embedding)

df_finqa = df_finqa_0[df_finqa_0['has_type2_question'] == True].reset_index(drop=True)


model_id = "openai/gpt-oss-20b" # 	qwen/qwen3-32b  openai/gpt-oss-20b llama-3.1-8b-instant
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            verbose=1)

# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")



class FinState(MessagesState):
    """ The state of the AI Assistant"""
    report_state: Annotated[bool, "Whether the financial report is provided by the user"]
    user_query: Annotated[List[str], "The list of questions from the user in continuous chat"]
    next_node: Annotated[Optional[str], "The next node to execute, either 'tool_node' or 'call_model_node'"]
    call_mode: Annotated[Literal['user','tool_call'], "The current mode of the call, either 'user' for model calls or 'tool_call' for tool executions"]
    context_content: Annotated[str, "The relevant content from the financial report to assist in answering the user's query"]
    context_examples: Annotated[Optional[str], "The relevant example conversations from the FinQA dataset to assist in answering the user's query"]
    message_out: Annotated[Optional[str], "The final answer to the user's query, if available"]
    full_response: Annotated[Optional[Any], "The full response from the model, including any tool calls or reasoning steps"]
    query_cont: Annotated[List[str], "Whether this is a new query or a continuation of the previous ones"]

def check_info_node(state: FinState):
    """Check if the financial report is provided by the user"""
    # Simple heuristic: if context_docs is empty or too short, we consider it insufficient
    if not state['report_state']:

        state['next_node'] = "END"
        state['message_out'] = "I'm sorry, but I don't have enough information to answer your question. Please provide the necessary financial documents."
    else:
        state['next_node'] = "call_model_node"

    # chekc if this is a new query or continuation
    prompt_template = PromptTemplate.from_template("This is the new query by the user: {new_query}. This is the previous queries by the user: {prev_queries}. Is the new query a continuation of the previous queries or a new one? Only Answer Yes or No.")

    new_query = state['user_query'][-1]
    prev_queries = '\n'.join(state['user_query'][:-1])

    chain = prompt_template|llm
    response = chain.invoke({"new_query": new_query,
                                "prev_queries": '\n'.join(prev_queries)})

    print('A continuation Question? ', response.content)
    if 'yes' in response.content.lower():
        state['query_cont'].append(new_query)
    else:
        state['query_cont'] = [new_query]

    print('connected queries: \n ', state['query_cont'])
    return {**state,
            'message_out': state['message_out'],
            'next_node': state['next_node'],
            'query_cont': state['query_cont'],
            "messages": []}

def should_continue_chat(state: FinState):
    """Determine whether to continue the chat or end the cycle."""
    # If there is no function call, then we finish
    if state['next_node']=="END":
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def tool_node(state: FinState):
    """Execute all tool calls from the last message in the state."""
    outputs = []
    for tool_call in state["full_response"].tool_calls:
        tool_name = tool_call['name']
        tool = tools_by_name[tool_name]
        tool_call_id = tool_call['id']
        tool_args = tool_call['args']
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
    if state['call_mode']=='user':
        query = state['user_query'][-1]
        queries = '\n'.join(state['query_cont'])
        relevant_docs = find_relevant_document_fast(embedding_model, queries, df_docs, 'content_embed', top_k=4)
        relevant_finq_docs = find_relevant_document_fast(embedding_model, queries, df_finqa, 'summary_emb', top_k=2)

        context_content = '\n'.join([doc for doc in relevant_docs['content']])
        context_examples = '\n'.join([f"Example {i}: {doc}" for i,doc in enumerate(relevant_finq_docs['conv_text_full'])]) 
        state['context_content'] = context_content
        state['context_examples'] = context_examples

        user_message = user_message_new
    else:
        user_message = user_message_tool
        query = state['user_query'][-1]
        context_content = state['context_content']
        context_examples = state['context_examples']

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        user_message
    ])

    model_react=chat_prompt|llm.bind_tools(tools)

    response = model_react.invoke({
        "context_content": context_content
        , "context_examples": context_examples
        , "chat_history": state["messages"]
        , "tool_names": ', '.join(tool_names)
        , "query": query
        })

    state['full_response'] = response
    if not response.tool_calls:
        state['call_mode'] = 'user'
        state['message_out'] = response.content
        out_message = [response.content]
    else:
        state['call_mode'] = 'tool_call'
        print(response.tool_calls)
        out_message = []



    return {**state,
            'call_mode': state['call_mode'],
            'context_content': state['context_content'],
            'context_examples': state['context_examples'],
            'message_out': state['message_out'],
            'full_response': state['full_response'],
            "messages": out_message}


def should_continue_react(state: FinState):
    """Determine whether to continue with tool use or end the ReAct loop."""
    call_mode = state['call_mode']
    # If there is no function call, then we finish
    if call_mode=='user':
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


builder = StateGraph(FinState)
builder.add_node("check_info_node", check_info_node)
builder.add_node("call_model_node", call_model_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "check_info_node")
builder.add_conditional_edges("check_info_node",
                              should_continue_chat, 
                              {"continue": "call_model_node",
                               "end": END})
builder.add_conditional_edges("call_model_node", 
                              should_continue_react, 
                              {"continue": "tool_node",
                               "end": END})
builder.add_edge("tool_node", "call_model_node")

finqa_graph = builder.compile(checkpointer=InMemorySaver())



class FinQAResponseAgent:
    """Async wrapper for running the finqa_graph ReAct agent as a reusable service."""

    def __init__(self, agent_graph, report_state: bool = False):
        """
        Initialize the response agent.

        Args:
            agent_graph: Compiled LangGraph instance (e.g., finqa_graph)
            report_state: Whether the financial report is provided by the user
        """
        self.agent = agent_graph
        self.report_state = report_state
        self.df_docs = None  # To be set after PDF upload

    def set_report_state(self, value: bool):
        """Update report_state dynamically (e.g., after PDF upload)."""
        self.report_state = value

    def set_documents(self, df_docs):
        """Attach parsed financial report (and optionally embedding model) to the agent."""
        self.df_docs = df_docs

    def convert_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
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
                "user_query": [user_input],
                "next_node": None,
                "call_mode": "user",
                "context_content": "",
                "context_examples": "",
                "message_out": None,
                "full_response": None,
            }
            state['messages'] = add_messages(state['messages'], HumanMessage(content=user_input))
        else:
            state = existing_state.values
            state['user_query'].append(user_input)
            state['report_state'] = self.report_state  # Ensure it's updated


        state['messages'] = add_messages(state['messages'], HumanMessage(content=user_input))

        return {"state": state, "run_config": run_config}

    async def predict(self, input_data: Dict[str, Any]) -> str:
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
                "full_response": agent_response['full_response']
        }


FinQAAgent = FinQAResponseAgent(finqa_graph)
