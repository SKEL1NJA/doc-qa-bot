import os
import streamlit as st
from extract import extract_text_from_pdf
from chunk import chunk_text
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"
COLLECTION_NAME = "documents"

SYSTEM_PROMPT = """You are a document QA assistant. Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, say "I don't have enough information in the document to answer that."
Do not use outside knowledge. Do not make up information.

Context:
{context}
"""


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def build_vector_store(pdf_path: str):
    client = chromadb.PersistentClient(path="chroma_db")
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(name=COLLECTION_NAME)

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    return collection


def retrieve_relevant_chunks(query: str, n_results: int = 3) -> list[str]:
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name=COLLECTION_NAME)
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["documents"][0]


def answer_question(query: str) -> str:
    chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(chunks)
    system_message = SYSTEM_PROMPT.format(context=context)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
    )
    return response["message"]["content"]


st.set_page_config(page_title="Document QA Bot")
st.title("Document QA Bot")

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    temp_path = os.path.join("temp_uploaded.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process document"):
        with st.spinner("Extracting, chunking, and embedding..."):
            build_vector_store(temp_path)
        st.session_state.vector_store_ready = True
        st.success("Document processed. Ask away.")

if st.session_state.vector_store_ready:
    query = st.text_input("Ask a question about the document")
    if query:
        with st.spinner("Thinking..."):
            answer = answer_question(query)
        st.write(answer)