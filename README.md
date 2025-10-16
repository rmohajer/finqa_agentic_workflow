# ConvFinQA Challenge

# ConvFinQA Challenge

Lightweight demo and tooling for conversational numerical reasoning over financial documents (ConvFinQA). This repo provides a Gradio-based app, dataset helpers, and agent scaffolding to parse PDFs, embed document chunks, and answer multi-turn finance questions.


DATASET: please see `dataset.md` for more information. We recommend you use this version of the data for the assignment, as it will save you a lot of time. If you have any questions, please don't hesitate to ask your point of contact. 



## Get started
### Prerequisites
- Python 3.12+

### Setup
1. Clone this repository
2. Use the UV environment manager to install dependencies:

```bash
# install uv
brew install uv

# set up env
uv sync

# add python package to env
uv add <package_name>
```

## Quick links
- Dataset description: [dataset.md](dataset.md)  
- Main entry: [src/main.py](src/main.py) — see [`src.main.generate_session_id`](src/main.py), [`src.main.handle_pdf_upload`](src/main.py), [`src.main.chat_with_pdf`](src/main.py)  
- Agent core: [src/agentic_system.py](src/agentic_system.py) — see [`agentic_system.FinQAResponseAgent`](src/agentic_system.py)  
- Parsing helpers: [src/parsing_helpers.py](src/parsing_helpers.py) — see [`parsing_helpers.fin_doc_parser`](src/parsing_helpers.py)  
- Project metadata: [pyproject.toml](pyproject.toml)  
- Example data: [data/convfinqa_dataset.json](data/convfinqa_dataset.json)  
- Notebooks: [notebooks/agentic_experiment.ipynb](notebooks/agentic_experiment.ipynb), [notebooks/data_parsing_experiment.ipynb](notebooks/data_parsing_experiment.ipynb)

## What this repo contains
- A Gradio UI app to upload PDFs, parse them into text/table chunks, compute embeddings, and run a FinQA-style agent over the parsed content. Entrypoint and UI logic live in [src/main.py](src/main.py).
- Agent and state management are implemented in [src/agentic_system.py](src/agentic_system.py) (the `FinQAResponseAgent` driven agent and related graph/state).
- Document parsing and embedding helpers in [src/parsing// filepath: README.md