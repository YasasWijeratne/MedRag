import os
import re
import tempfile

import fitz

MAX_FILE_SIZE = 5 * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "", filename)


def validate_pdf(uploaded_file) -> str:
    file_size = len(uploaded_file.getvalue())

    if file_size > MAX_FILE_SIZE:
        raise ValueError("File exceeds the maximum allowed size of 5 MB.")

    header = uploaded_file.read(4)

    if header != b"%PDF":
        raise ValueError("Only valid PDF files are allowed.")

    uploaded_file.seek(0)

    return sanitize_filename(uploaded_file.name)


def extract_pdf_text(uploaded_file):
    source_name = validate_pdf(uploaded_file)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        document = fitz.open(temp_path)

        pages = []

        for page_number in range(len(document)):
            page = document.load_page(page_number)

            page_text = page.get_text()

            if not page_text.strip():
                continue

            pages.append(
                {
                    "source": source_name,
                    "page": page_number + 1,
                    "text": page_text
                }
            )

        document.close()

        return pages

    except Exception as error:
        print(f"PDF processing error: {error}")

        raise ValueError(
            "Unable to process the uploaded PDF."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)