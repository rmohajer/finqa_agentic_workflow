#%%
"""
Main typer app for ConvFinQA
"""
import os, sys
from rich import print as rich_print

# Load .env FIRST, before any Google Cloud imports
from dotenv import load_dotenv
load_dotenv()

# Monkey patch for gradio_client bug: TypeError: argument of type 'bool' is not iterable
# Bug: get_type() and _json_schema_to_python_type() receive boolean schema instead of dict
import gradio_client.utils as gc_utils

_original_get_type = gc_utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _original_get_type(schema)
gc_utils.get_type = _patched_get_type

_original_json_schema_to_python_type = gc_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)
gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

import gradio as gr
import uuid
import pandas as pd
import asyncio

from src.agentic_system import FinState, finqa_graph, FinQAResponseAgent, min_chunk_length, _ensure_finqa_embeddings
from src.parsing_helpers import fin_doc_parser, parse_embedding

import shutil
import os

paths = [
    os.path.expanduser("~/.cache/huggingface"),
    os.path.expanduser("~/.cache/pip"),
    "/home/user/app/__pycache__",
]

for path in paths:
    shutil.rmtree(path, ignore_errors=True)

print("🧹 Cache cleaned successfully!")

# Pre-compute/load ConvFinQA embeddings at startup (before building app)
print("🔄 Initializing ConvFinQA embeddings...")
_ensure_finqa_embeddings()
print("✅ ConvFinQA embeddings ready!")

def generate_session_id():
    """Generate a unique session ID for each user."""
    return str(uuid.uuid4())


#%%
# Global persistent FinQA agent instance
FinQAAgent = FinQAResponseAgent(finqa_graph)

async def handle_pdf_upload(file_obj, session_id, parser='docling'):
    """Asynchronously parse PDF and store text embeddings."""
    from src.agentic_system import embedding_model
    return await FinQAAgent.handle_pdf_upload(file_obj, session_id, min_chunk_length, embedding_model, parser)

# ---------------------------
# Chat Function
# ---------------------------
async def chat_with_pdf(user_input, session_id):
    """Chat with the parsed financial document."""
    session_data = FinQAAgent._get_session_data(session_id)
    report_state = session_data.get("report_state", False)
    df_docs = session_data.get("df_docs", None)
    
    if not report_state or df_docs is None:
        return "⚠️ Please upload and parse a PDF file first."

    input_to_agent = {
        "user_id": session_id,
        "user_input": user_input,
    }

    # Run FinQA agent asynchronously
    response = await FinQAAgent.predict(input_to_agent)
    return response['output']

# ---------------------------
# Gradio App 
# ---------------------------
with gr.Blocks(fill_height=True, title="📊 Chat with Financial PDF", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 📄 Chat with Your Financial Report
        1️⃣ Upload a PDF file containing a financial report.  
        2️⃣ Ask questions about it in the chat box.  
        3️⃣ The AI will respond using its internal FinQA reasoning graph.
        """
    )

    # Persistent session ID (simple string, no schema issues)
    session_id = gr.State(generate_session_id)

    # File Upload
    with gr.Row():
        pdf_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
        parser_select = gr.Dropdown(
            choices=["docling", "doc_ai"],
            value="docling",
            label="Parser",
            info="docling: local (CPU), doc_ai: Google Document AI (cloud)"
        )
        parse_btn = gr.Button("Parse PDF", variant="primary")

    # Chat Interface
    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="Ask something about the PDF...")
        ai_output = gr.Textbox(label="AI Response", interactive=False, lines=20)

    with gr.Row():
        chat_btn = gr.Button("Send Question")
        clear_btn = gr.Button("Clear Chat", variant="secondary")

    # Status display
    status_output = gr.Textbox(label="Status", interactive=False, lines=2)

    # ---------------------------
    # Button Interactions
    # ---------------------------
    parse_btn.click(
        fn=handle_pdf_upload,
        inputs=[pdf_input, session_id, parser_select],
        outputs=[status_output],
    )

    chat_btn.click(
        fn=chat_with_pdf,
        inputs=[user_input, session_id],
        outputs=[ai_output],
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[user_input, ai_output],
    )


if __name__ == "__main__":
    rich_print("[green]🚀 Launching Gradio app...[/green]")
    app.queue()  # enables async support
    app.launch(share=False, inline=False, height=900, server_name="0.0.0.0", server_port=7860)