import re


def clean_text(input_file='extracted_text.txt'):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove extra whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'Page \d+', '', text)

    # Remove chapter headers if repetitive
    text = re.sub(r'\nChapter \d+\n', '\n', text)

    # Clean up
    text = text.strip()

    # Save cleaned text
    with open('cleaned_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Cleaned text: {len(text)} characters")
    return text


# Usage
cleaned_text = clean_text()