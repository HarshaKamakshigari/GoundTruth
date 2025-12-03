from fastapi import FastAPI
from .rag import RAGPipeline
from .llm import fake_llm_answer

app = FastAPI()
rag = RAGPipeline()

@app.on_event("startup")
def load_data():
    rag.load_data()

@app.get("/ask")
def ask_bot(query: str):
    context = rag.query(query)
    answer = fake_llm_answer(query, context)
    return {"response": answer}

