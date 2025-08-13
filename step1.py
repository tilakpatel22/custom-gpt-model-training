import PyPDF2


def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

    # Save extracted text
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Extracted {len(text)} characters")
    return text


# Usage
pdf_path = "test.pdf"  # Replace with your PDF path
extracted_text = extract_pdf_text(pdf_path)