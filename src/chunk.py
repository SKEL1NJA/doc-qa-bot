from langchain_text_splitters import RecursiveCharacterTextSplitter
from extract import extract_text_from_pdf


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    print("--- Chunk 1 ---")
    print(chunks[0])
    print("--- Chunk 2 ---")
    print(chunks[1])