from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

def load_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found.\n\n"
            "Create a .env file and add:\n\n"
            "GROQ_API_KEY=your_api_key"
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
    )

    return llm