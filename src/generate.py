import ollama
from retrieve import retrieve_relevant_chunks

CHAT_MODEL = "llama3.1"

SYSTEM_PROMPT = """You are a document QA assistant. Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, say "I don't have enough information in the document to answer that."
Do not use outside knowledge. Do not make up information.

Context:
{context}
"""


def answer_question(query: str) -> str:
    chunks = retrieve_relevant_chunks(query, n_results=3)
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


if __name__ == "__main__":
    query = "What is the duration of the internship?"
    answer = answer_question(query)
    print(f"Question: {query}")
    print(f"Answer: {answer}")