import os
import streamlit as st
from extract import extract_text_from_pdf
from chunk import chunk_text
from llm_provider import get_embedding, chat
import chromadb

COLLECTION_NAME = "documents"

SYSTEM_PROMPT = """You are a document QA assistant. Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, say "I don't have enough information in the document to answer that."
Do not use outside knowledge. Do not make up information.

Context:
{context}
"""


def build_vector_store(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No extractable text found in this PDF. It may be a scanned image without a text layer.")

    client = chromadb.PersistentClient(path="chroma_db")
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(name=COLLECTION_NAME)

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


def answer_question(query: str) -> tuple[str, list[str]]:
    chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(chunks)
    system_message = SYSTEM_PROMPT.format(context=context)
    answer = chat(system_message, query)
    return answer, chunks


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
        try:
            with st.spinner("Extracting, chunking, and embedding..."):
                build_vector_store(temp_path)
            st.session_state.vector_store_ready = True
            st.success("Document processed. Ask away.")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to process document: {e}")

if st.session_state.vector_store_ready:
    query = st.text_input("Ask a question about the document")
    if query:
        try:
            with st.spinner("Thinking..."):
                answer, sources = answer_question(query)
            st.write(answer)

            with st.expander("View source chunks used"):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk)
        except Exception as e:
            st.error(f"Something went wrong while answering: {e}")
            st.info("If using Ollama, make sure it's running. If using Gemini, check your API key.")