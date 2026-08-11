from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    full_text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:  # some pages (e.g. pure images) may return None
            full_text += page_text + "\n"
        else:
            print(f"Warning: no extractable text on page {page_number}")

    return full_text


if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    print(f"Extracted {len(text)} characters.")
    print("--- First 500 characters ---")
    print(text[:500])