from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from llm import load_llm


# -----------------------------
# Retriever
# -----------------------------

def get_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


# -----------------------------
# Question Answering Chain
# -----------------------------

def get_qa_chain(retriever):

    llm = load_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are an intelligent AI Research Assistant.

Previous Conversation:
{history}

Use ONLY the following document context to answer.

Context:
{context}

Question:
{input}

Rules:
1. Never make up information.
2. If the answer is not found in the context, reply:
"I couldn't find that information in the uploaded PDF."
3. Answer clearly.
4. Use bullet points whenever appropriate.
5. Keep answers concise.

Answer:
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


# -----------------------------
# Summary Chain
# -----------------------------

def get_summary_chain():

    llm = load_llm()

    summary_prompt = ChatPromptTemplate.from_template(
        """
You are an expert document summarizer.

Summarize the following document.

Document:
{context}

Your summary must contain:

• Title

• Main Topics

• Important Concepts

• Key Points

• Conclusion
"""
    )

    summary_chain = create_stuff_documents_chain(
        llm,
        summary_prompt
    )

    return summary_chain