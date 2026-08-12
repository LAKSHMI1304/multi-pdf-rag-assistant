import streamlit as st
import os
import traceback
import time

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import load_pdf
from text_splitter import split_documents
from embeddings import get_embeddings
from vector_store import create_vector_store
from rag import get_retriever, get_qa_chain, get_summary_chain

# ---------------------------------
# PDF Export Function (Step 3)
# ---------------------------------

def create_chat_pdf(messages):

    pdf_file = "chat_history.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "<b>Multi-PDF Research Assistant Chat</b>",
            styles["Title"]
        )
    )

    story.append(Paragraph("<br/><br/>", styles["Normal"]))

    for message in messages:

        role = message["role"].capitalize()

        content = message["content"]

        story.append(
            Paragraph(
                f"<b>{role}:</b> {content}",
                styles["BodyText"]
            )
        )

        story.append(Paragraph("<br/>", styles["Normal"]))

    doc.build(story)

    return pdf_file

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="📚 Multi-PDF Intelligent Research Assistant",
    layout="wide"
)
# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="📚 Multi-PDF Intelligent Research Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("📚 Multi-PDF Intelligent Research Assistant")

st.markdown("""
Upload one or more PDF files and ask questions based only on the uploaded documents.
""")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📊 Dashboard")

st.sidebar.info("""
### Multi-PDF Research Assistant

Powered By

- 🤖 Groq
- 🧠 HuggingFace
- 🗄️ FAISS
- 🔗 LangChain
""")
# -----------------------------
# Clear Chat Button
# -----------------------------

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# -----------------------------
# Chat Memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Upload PDFs
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs("data/pdfs", exist_ok=True)

    all_docs = []

    progress = st.progress(0)

    # Save PDFs

    for index, uploaded_file in enumerate(uploaded_files):

        save_path = os.path.join(
            "data/pdfs",
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")

        docs = load_pdf(save_path)

        all_docs.extend(docs)

        progress.progress((index + 1) / len(uploaded_files))

    st.success(f"Loaded {len(all_docs)} pages.")
    st.subheader("📄 File Information")

    for uploaded_file in uploaded_files:

        file_size = round(uploaded_file.size / 1024, 2)

        st.info(
        f"""
    📄 File Name : {uploaded_file.name}

    📦 Size : {file_size} KB

    ✅ Status : Successfully Processed
    """
    )

    st.sidebar.metric(
        "📄 Total Pages",
        len(all_docs)
    )

    # -----------------------------
    # Split Documents
    # -----------------------------

    chunks = split_documents(all_docs)

    st.success(f"Created {len(chunks)} text chunks.")

    st.sidebar.metric(
        "🧩 Text Chunks",
        len(chunks)
    )

    if len(chunks) == 0:

        st.error("No readable text found in the uploaded PDFs.")

        st.stop()

    try:

        # -----------------------------
        # Embeddings
        # -----------------------------

        with st.spinner("Loading HuggingFace Embeddings..."):

            embeddings = get_embeddings()

        # -----------------------------
        # Vector Store
        # -----------------------------

        with st.spinner("Creating FAISS Vector Database..."):

            vectorstore = create_vector_store(
                chunks,
                embeddings
            )
        st.success("📂 FAISS Database Loaded")
        # -----------------------------
        # Retriever
        # -----------------------------

        retriever = get_retriever(vectorstore)

        # -----------------------------
        # QA Chain
        # -----------------------------

        qa_chain = get_qa_chain(retriever)
        summary_chain = get_summary_chain()

        st.success("✅ AI Assistant Ready!")

        st.sidebar.success("🟢 System Ready")

        st.sidebar.metric("🤖 LLM", "Groq")
        st.sidebar.metric("🗄️ Vector DB", "FAISS")
        st.sidebar.metric("🧠 Embedding", "MiniLM")
        # -----------------------------
        # Generate PDF Summary
        # -----------------------------

        if st.button("📄 Generate PDF Summary"):

            with st.spinner("Generating Summary..."):

                summary = summary_chain.invoke(
                   {
                        "context": chunks
                    }
                )

            # Handle different LangChain return types
            summary_text = (
                summary if isinstance(summary, str)
                else summary.get("answer", str(summary))
            )

            st.divider()

            st.subheader("📄 AI Generated Summary")

            st.write(summary_text)

            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name="pdf_summary.txt",
                mime="text/plain"
            )
        # -----------------------------
        # Chat Input
        # -----------------------------
        
        question = st.chat_input(
            "Ask anything about your PDFs..."
        )

        if question:
            # Save user message
            st.session_state.messages.append(
                {
                     "role": "user",
                    "content": question
                }
            )
            with st.chat_message("user"):
                    st.markdown(question)
            # Search documents
            start_time = time.time()
            with st.spinner("Searching documents..."):
                history="\n".join(
                    [
                        f"{m['role']}: {m['content']}"
                        for m in st.session_state.messages
                    ]
                )
                response = qa_chain.invoke(
                    {
                        "input": question
                    },
                    config={
                        "configurable": {
                            "session_id": "default"
                        }
                    }
                )

            end_time = time.time()

            response_time = round(end_time - start_time,2)  
            answer = response["answer"]
            # Save assistant message
            st.session_state.messages.append(
               {
                  "role": "assistant",
                  "content": answer
                }
            )
            #Display assistant answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"⏱ Response Time: {response_time} seconds")
            # Sidebar Metrics
            retrieved_chunks = len(response.get("context", []))
            st.sidebar.metric(
                "📑 Retrieved Chunks",
                retrieved_chunks
            ) 
            st.sidebar.metric(
                "⏱ Response Time",
                f"{response_time} sec"
            ) 

            # Retrieved Sources
            if "context" in response:
                st.divider()
                st.subheader("📄 Retrieved Sources")

                for i, doc in enumerate(response["context"]):

                    page = doc.metadata.get("page", 0) + 1

                    source = os.path.basename(doc.metadata.get("source", "Unknown"))

                    with st.expander( f"Source {i+1} | {source} | Page {page}"):
                        st.write(doc.page_content)

            # Download Button
            st.divider()

            st.download_button(
                label="📄 Download Answer (.txt)",
                data=answer,
                file_name="answer.txt",
                mime="text/plain"
            )
            # -----------------------------
            # Export Entire Chat as PDF
            # -----------------------------

            if st.button("📥 Export Chat as PDF"):

                pdf_path = create_chat_pdf(st.session_state.messages)

                with open(pdf_path, "rb") as pdf_file:

                    st.download_button(
                        label="⬇ Download Chat PDF",
                        data=pdf_file,
                        file_name="Chat_History.pdf",
                        mime="application/pdf"
                    )
          

            
    except Exception as e:

        st.error("❌ Error Occurred")

        st.code(str(e))

        st.code(traceback.format_exc())