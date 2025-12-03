import os
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# Configure Gemini client
genai.configure(api_key=API_KEY)

# Initialize the model once
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_answer(question: str, context: str) -> str:
    """
    Uses Gemini 2.5 Flash to generate a clean, conversational answer
    based on retrieved RAG context.
    """

    prompt = f"""
You are a hyper-personalized customer support agent for Starbucks.

Use ONLY the following context to answer the user's question.
If the context includes store information, offers, timings, or user coupons,
use them to give a specific, relevant, natural response.

If something is not in the context, do not hallucinate.
Be concise, friendly, and helpful.

--------------------
CONTEXT:
{context}
--------------------

USER QUESTION:
{question}

Now generate the best possible answer:
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"⚠ Gemini error: {e}"