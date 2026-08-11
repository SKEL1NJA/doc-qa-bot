import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "documents"


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def retrieve_relevant_chunks(query: str, n_results: int = 3) -> list[str]:
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results["documents"][0]


if __name__ == "__main__":
    query = "What is the duration of the internship?"
    chunks = retrieve_relevant_chunks(query)

    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Result {i} ---")
        print(chunk)
        print()