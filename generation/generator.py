import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables."
    )

client = Groq(api_key=API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"


def generate_answer(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        answer = response.choices[0].message.content

        usage = response.usage

        return {
            "answer": answer,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }

    except Exception:
        print("Generation error occurred.")

        raise ValueError(
            "Unable to generate an answer at this time."
        )