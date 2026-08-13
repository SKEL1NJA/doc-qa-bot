from dotenv import load_dotenv
load_dotenv()

import os
import ollama
import google.generativeai as genai

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "llama3.1"

GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-3.6-flash"

if PROVIDER == "gemini":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)


def get_embedding(text: str) -> list[float]:
    if PROVIDER == "gemini":
        result = genai.embed_content(model=GEMINI_EMBED_MODEL, content=text)
        return result["embedding"]
    else:
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]


def chat(system_message: str, user_message: str) -> str:
    if PROVIDER == "gemini":
        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=system_message
        )
        response = model.generate_content(user_message)
        return response.text
    else:
        response = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response["message"]["content"]