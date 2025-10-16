
from typing import (List, Annotated,Sequence,TypedDict,Any,Optional,Dict,Union,Literal)
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
# prompt templates


class model_call_output(BaseModel):
    """Output schema for the model call."""    
    Thought: List[str] = Field(description= "the AI assistant's thought process, step by step") 
    References: Optional[List[str]] = Field(description="list of references the AI has used from the context to support the reasoning and final answer")
    Final_answer: Optional[str] = Field(description="the final answer to the user's question, based on the thought process and any tool results")


system_msg_template = """You are a financial expert AI assistant. You have access to the following tools: {tool_names}. You are also given relevant contextual information to answer the question, together with example answers on similar questions that could help your thinking process. 

When you need to perform a calculation, use the appropriate tool. Always think step by step and explain your reasoning before providing an answer. Always explain your thinking process to help users understand your approach.

When responding to queries:
1. First, make sure that you respond based on the chat history and the contextual information provided by the user
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

"""

user_msg_new_template = """

# The chat history with the user is as follows delimited by CHAT_HISTORY.

Chat History:

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

# Question from the user: {query}

# You are given the following context to help you answer the question: 

<CONTEXT>
{context_content}
</CONTEXT>

# Example thinking process on similar conversational questions:

<EXAMPLES>
{context_examples}  
</EXAMPLES>


# You have access to the following tools, delimited by TOOLS: 

<TOOLS>
{tool_names}
</TOOLS>

Please answer the user's Question based on the CHAT_HISTORY and CONTEXT. Remember to think step by step and use the TOOLS when necessary. You can inspire by the EXAMPLES provided to plan your steps.
Provide a final answer at the end.
If you don't know the answer, just say you don't know. Do not try to make up an answer.

# If you are producing the final answer please give it in the following format (if you are going to user other tools please skip this step):

Thought: [the AI assistant's thought process, step by step]

References: [list of references the AI has used from the context to support the reasoning and final answer]

Final_answer: [the final answer to the user's question, based on the thought process and any tool results]

"""


user_msg_tool_template = """

# Question from the user: {query}

# You are given the following context to help you answer the question: 

<CONTEXT>
{context_content}
</CONTEXT>

# You have access to the following tools, delimited by TOOLS: 

<TOOLS>
{tool_names}
</TOOLS>

# The chat history with the user so far delimited by CHAT_HISTORY.

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

Please continue answering the user's query based on the CHAT_HISTORY and CONTEXT. Remember to think step by step and use the TOOLS when necessary.


# If you are producing the final answer please give it in the following format (if you are going to user other tools please skip this step):

Thought: [the AI assistant's thought process, step by step]

References: [list of references the AI has used from the context to support the reasoning and final answer]

Final_answer: [the final answer to the user's question, based on the thought process and any tool results]

"""


system_message = SystemMessagePromptTemplate.from_template(system_msg_template)
user_message_new = HumanMessagePromptTemplate.from_template(user_msg_new_template) 
user_message_tool = HumanMessagePromptTemplate.from_template(user_msg_tool_template) 

tool_flag = False

if tool_flag:
    user_message = user_message_tool
else:
    user_message = user_message_new

chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    user_message
])