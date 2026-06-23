SYSTEM_PROMPT = """
You are MedRAG, a medical literature assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- If the answer is not contained in the context, say so clearly.
- Cite factual statements using this format:
  (Source: filename, Page X)
- If multiple sources discuss the topic differently, mention the difference and cite the relevant sources.
- Keep answers clear, accurate, and concise.
""".strip()


def build_prompt(query, results):
    context_sections = []

    for result in results:
        context_sections.append(
            f"""[Source: {result["source"]} | Page: {result["page"]}]
{result["text"]}"""
        )

    context = "\n\n".join(context_sections)

    prompt = f"""
{SYSTEM_PROMPT}

Context:

{context}

Question:
{query}

Answer:
""".strip()

    return prompt