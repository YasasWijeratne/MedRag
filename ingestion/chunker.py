CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_pages(pages):
    chunks = []

    for page_data in pages:
        source = page_data["source"]
        page = page_data["page"]
        text = page_data["text"].strip()

        if not text:
            continue

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "text": chunk_text
                    }
                )

            start += CHUNK_SIZE - CHUNK_OVERLAP
            chunk_id += 1

    return chunks