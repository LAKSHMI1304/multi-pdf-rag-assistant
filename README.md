# 📚 Multi-PDF Intelligent Research Assistant

An AI-powered application that allows users to upload multiple PDF documents and ask questions using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- 📄 Upload multiple PDF files
- 🤖 Ask questions in natural language
- 🔍 Semantic search using FAISS
- 🧠 HuggingFace Embeddings
- ⚡ Groq LLM integration
- 💬 Chat history
- 📄 AI-generated PDF summary
- 📥 Download answers
- 📊 Dashboard with document statistics

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace
- Groq API

## 📂 Project Structure

```
app.py
config.py
llm.py
rag.py
embeddings.py
vector_store.py
pdf_processor.py
text_splitter.py
requirements.txt
```

## Installation

```bash
git clone https://github.com/LAKSHMI1304/multi-pdf-rag-assistant.git
cd multi-pdf-rag-assistant

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

## Author

Lakshmi
