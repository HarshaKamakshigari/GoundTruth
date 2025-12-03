"""
One-Store Hyper Personalized Customer Support Agent
Store: Starbucks SRM Coffee Point
"""

import os
import re
import json
import math
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from geopy.distance import geodesic
import google.generativeai as genai

load_dotenv()

# -----------------------------
# CONSTANTS (ONE STORE)
# -----------------------------
STORE_FILE = "data/store.json"
STATUS_FILE = "data/store_status.json"
PROFILE_FILE = "data/profile.txt"
INDEX_FILE = "rag/profile.index"
CHUNKS_FILE = "rag/chunks.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# -----------------------------
# Load store info
# -----------------------------
store = json.load(open(STORE_FILE, "r"))
store_status = json.load(open(STATUS_FILE, "r"))

os.makedirs("rag", exist_ok=True)

# -----------------------------
# FastAPI & CORS
# -----------------------------
app = FastAPI(title="One Store Support Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# -----------------------------
# Conversation Memory
# -----------------------------
memory_db: Dict[str, List[Dict[str, str]]] = {}

def save_memory(user: str, role: str, text: str):
    mem = memory_db.setdefault(user, [])
    mem.append({"role": role, "text": text})
    if len(mem) > 10:
        memory_db[user] = mem[-10:]

def memory_to_text(user: str):
    mem = memory_db.get(user, [])
    text = ""
    for m in mem:
        text += f"{m['role'].upper()}: {m['text']}\n"
    return text

# -----------------------------
# Privacy Masking
# -----------------------------
def mask(text: str):
    text = re.sub(r"\b\d{10}\b", "[PHONE_MASKED]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z]+\.[A-Za-z]+", "[EMAIL_MASKED]", text)
    return text

# -----------------------------
# RAG Build
# -----------------------------
embed_model = None

def get_embed():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model

def chunk_text(text: str, size=350, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def build_rag(text: str):
    chunks = chunk_text(text)
    model = get_embed()
    emb = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(emb.astype("float32"))
    faiss.write_index(index, INDEX_FILE)
    json.dump(chunks, open(CHUNKS_FILE, "w"))
    return len(chunks)

def rag_search(query: str, k=3):
    if not os.path.exists(INDEX_FILE):
        return []
    index = faiss.read_index(INDEX_FILE)
    chunks = json.load(open(CHUNKS_FILE, "r"))
    model = get_embed()
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype("float32"), k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# Distance Calculation
# -----------------------------
def calculate_distance(lat, lon):
    try:
        return int(geodesic((lat, lon), (store["lat"], store["lon"])).meters)
    except:
        return None

# -----------------------------
# Gemini Integration
# -----------------------------
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

def gemini(prompt):
    if not API_KEY:
        return None
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content(prompt)
        return out.text
    except Exception as e:
        print("Gemini error:", e)
        return None

# -----------------------------
# API Models
# -----------------------------
class ChatReq(BaseModel):
    message: str
    user_id: str = "user"
    lat: float = 17.445
    lon: float = 78.391

class ChatRes(BaseModel):
    reply: str
    store_info: dict
    personalized: bool
    features_used: list

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"msg": "One Store Support Bot Running"}

@app.post("/api/upload_profile")
async def upload_profile(file: UploadFile = File(...)):
    """
    Upload a customer PDF or TXT and rebuild RAG.
    """
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        pdf = PdfReader(file.file)
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = (await file.read()).decode()

    open(PROFILE_FILE, "w").write(text)
    chunks = build_rag(text)
    return {"indexed_chunks": chunks}

@app.post("/api/chat", response_model=ChatRes)
async def chat(req: ChatReq):
    user_msg = mask(req.message)
    save_memory(req.user_id, "user", user_msg)

    # RAG
    rag_ctx = rag_search(user_msg)

    # Distance
    dist = calculate_distance(req.lat, req.lon)

    # Store status
    inv = store_status.get("inventory", {})
    open_now = store_status.get("open_now", True)

    # RAG + memory + store fused prompt
    prompt = f"""
User: {user_msg}

Store: {store['name']}
Address: {store['address']}
Distance: {dist} meters
Offer: {store['offer']}
Open now: {open_now}

Inventory: {json.dumps(inv)}

RAG Context:
{rag_ctx}

Conversation Memory:
{memory_to_text(req.user_id)}

Task:
- Reply warmly, use customer personalization from RAG.
- If they are cold → recommend Hot Cocoa.
- If they ask about stock → check inventory.
- Keep reply short and friendly.
"""

    # Gemini or fallback
    reply = gemini(prompt)
    if not reply:
        # Fallback logic
        if "cold" in user_msg.lower():
            reply = f"You’re {dist}m away from Starbucks SRM Coffee Point! Come inside — we have Hot Cocoa and your 10% coupon applies."
        else:
            reply = f"The Starbucks SRM Coffee Point is {dist}m away. Offer: {store['offer']}."

    save_memory(req.user_id, "assistant", reply)

    return ChatRes(
        reply=reply,
        personalized=True,
        store_info={
            "name": store["name"],
            "distance": dist,
            "open_now": open_now,
            "inventory": inv
        },
        features_used=["RAG", "Location", "Gemini", "Inventory"]
    )

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
