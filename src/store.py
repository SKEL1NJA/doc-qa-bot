import chromadb
import ollama
from extract import extract_text_from_pdf
from chunk import chunk_text

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "documents"


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def build_vector_store(pdf_path: str):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )

    return collection


if __name__ == "__main__":
    collection = build_vector_store("sample.pdf")
    print(f"Stored {collection.count()} chunks in ChromaDB.")