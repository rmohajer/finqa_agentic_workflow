#%%
"""
Main typer app for ConvFinQA
"""
import os, sys
from rich import print as rich_print
import gradio as gr
import uuid
import pandas as pd
import asyncio

from logger import get_logger
from  agentic_system import FinState, finqa_graph, FinQAResponseAgent, embedding_model
from parsing_helpers import fin_doc_parser, parse_embedding


min_chunk_length=800

def generate_session_id():
    """Generate a unique session ID for each user."""
    return str(uuid.uuid4())

#%%

# ---------------------------
# Core logic
# ---------------------------
# Global persistent FinQA agent instance
FinQAAgent = FinQAResponseAgent(finqa_graph)

async def handle_pdf_upload(file_obj, embedding_model=embedding_model, min_chunk_length=800):
    """Asynchronously parse PDF and store text embeddings."""
    if file_obj is None:
        FinQAAgent.set_report_state(False)
        return None, False, "❌ Please upload a PDF file first."

    try:
        # Assuming fin_doc_parser can work sync (wrap it in asyncio.to_thread if not)
        df_docs, docs = await asyncio.to_thread(
            fin_doc_parser, file_obj, embedding_model, min_chunk_length
        )

        if df_docs is None or df_docs.empty:
            FinQAAgent.set_report_state(False)
            return df_docs, False, "⚠️ No readable text found in the PDF."
        else:
            FinQAAgent.set_report_state(True)
            FinQAAgent.set_documents(df_docs)

        return df_docs, True, "✅ PDF parsed successfully!"
    except Exception as e:
        FinQAAgent.set_report_state(False)
        return None, False,  f"❌ Error during PDF parsing: {str(e)}"

# ---------------------------
# Chat Function
# ---------------------------
async def chat_with_pdf(user_input, report_state, df_docs, session_id):
    """Chat with the parsed financial document."""
    if not report_state or df_docs is None:
        return "⚠️ Please upload and parse a PDF file first."

    input_to_agent = {
        "user_id": session_id,
        "user_input": user_input,
    }

    # Run FinQA agent asynchronously
    response = await FinQAAgent.predict(input_to_agent)
    # FinQAResponseAgent.predict() returns message_out (string)
    return response['output']

# ---------------------------
# Gradio App
# ---------------------------
with gr.Blocks(title="📊 Chat with Financial PDF") as app:
    gr.Markdown(
        """
        # 📄 Chat with Your Financial Report
        1️⃣ Upload a PDF file containing a financial report.  
        2️⃣ Ask questions about it in the chat box.  
        3️⃣ The AI will respond using its internal FinQA reasoning graph.
        """
    )

    # Persistent session ID
    session_id = gr.State(generate_session_id())

    # File Upload
    with gr.Row():
        pdf_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
        parse_btn = gr.Button("Parse PDF", variant="primary")

    # Chat Interface
    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="Ask something about the PDF...")
        ai_output = gr.Textbox(label="AI Response", interactive=False)

    with gr.Row():
        chat_btn = gr.Button("Send Question")
        clear_btn = gr.Button("Clear Chat", variant="secondary")

    # States
    df_docs = gr.State(None)
    report_state = gr.State(False)

    # ---------------------------
    # Button Interactions
    # ---------------------------
    parse_btn.click(
        fn=handle_pdf_upload,
        inputs=[pdf_input],
        outputs=[df_docs,report_state, ai_output],
    )

    chat_btn.click(
        fn=chat_with_pdf,
        inputs=[user_input, report_state, df_docs, session_id],
        outputs=ai_output,
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[user_input, ai_output],
    )


if __name__ == "__main__":
    rich_print("[green]🚀 Launching Gradio app...[/green]")
    app.queue()  # enables async support
    app.launch(share=True)
