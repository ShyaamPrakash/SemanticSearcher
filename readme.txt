# 🤖 NotesBot

NotesBot is a simple RAG chatbot that lets you upload PDFs and TXT files and ask questions about them.

It uses semantic search to find the most relevant parts of your documents and then uses an LLM (groq) to generate answers based on that context.

## Features

- Upload PDF and TXT files
- Generate embeddings using Sentence Transformers
- Semantic search with cosine similarity
- Chat interface built with Streamlit
- Source citations with file and page numbers
- Recent history based answers
- Basic corpus analytics
- Revise mode with a lot of different options where the embedded data is sent to api to get meaningful results
- Increased overall function of the app through api calls since without those the app is less useful
- Different modes for the model behaviour


## Tech Stack
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Scikit-learn
- pdfplumber
- Groq API

## How it Works

Upload Documents -> Generate Embeddings -> Ask a Question -> Retrieve Relevant Chunks -> Generate Answer


## Running Locally

git clone <repo-url>
cd NotesBot

pip install -r requirements.txt

streamlit run app.py


Add your API key in:


.streamlit/secrets.toml


Example:


GROQ_API_KEY = "your_api_key"


## Reason for building
    This project enabled me to get a better grasp on simple RAG and embdding and api calls.

Built by Shyaam
