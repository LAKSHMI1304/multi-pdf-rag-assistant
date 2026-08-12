import os
import hashlib

from langchain_community.vectorstores import FAISS

FAISS_PATH = "faiss_index"
HASH_FILE = "faiss_index/pdf_hash.txt"


def get_pdf_hash(chunks):
    """
    Generate a hash based on PDF contents.
    """

    text = ""

    for chunk in chunks:
        text += chunk.page_content

    return hashlib.md5(text.encode()).hexdigest()


def create_vector_store(chunks, embeddings):

    current_hash = get_pdf_hash(chunks)

    # Check if FAISS index already exists
    if os.path.exists(FAISS_PATH) and os.path.exists(HASH_FILE):

        with open(HASH_FILE, "r") as f:
            saved_hash = f.read()

        if saved_hash == current_hash:

            print("Loading existing FAISS index...")

            return FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

    # Create a new FAISS index
    print("Creating new FAISS index...")

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local(FAISS_PATH)

    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)

    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    return vector_db