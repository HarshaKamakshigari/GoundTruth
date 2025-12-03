from fastapi import FastAPI, Query
from pydantic import BaseModel

from .rag import RAGPipeline
from .llm import generate_answer


app = FastAPI(title="Starbucks Hyper-Personalized RAG")
rag = RAGPipeline()


class AskRequest(BaseModel):
    question: str
    user_id: str | None = None  
    store_id: str | None = None 


@app.on_event("startup")
def startup_event():
    rag.load_data()


@app.post("/ask")
def ask(req: AskRequest):
    
    context = rag.build_context(
        req.question,
        user_id=req.user_id,
        store_id=req.store_id,
        k=5,
    )

    print("Retrieved CONTEXT:")
    print(context)
    answer = generate_answer(req.question, context)
    return {
        "question": req.question,
        "user_id": req.user_id,
        "store_id": req.store_id,
        "answer": answer,
        "context": context,
    }



@app.get("/")
def root():
    return {"message": "Starbucks RAG API is running. Use POST /ask with {question: ...}"}