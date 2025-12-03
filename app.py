"""
One-Store Hyper Personalized Customer Support Agent
Store: Starbucks SRM Coffee Point
"""

import os
import re
import json
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from geopy.distance import geodesic
import google.generativeai as genai

# -------------------------------------------------------------------------
# ENV LOADING (FORCE LOAD .env)
# -------------------------------------------------------------------------
load_dotenv(dotenv_path=".env", override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
print("DEBUG: Loaded Gemini key →", API_KEY)  # MUST show your key

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
STORE_FILE = "data/store.json"
STATUS_FILE = "data/store_status.json"
PROFILE_FILE = "data/profile.txt"
INDEX_FILE = "rag/profile.index"
CHUNKS_FILE = "rag/chunks.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

os.makedirs("rag", exist_ok=True)

# -------------------------------------------------------------------------
# LOAD STORE DATA
# -------------------------------------------------------------------------
store = json.load(open(STORE_FILE))
store_status = json.load(open(STATUS_FILE))

# -------------------------------------------------------------------------
# APP
# -------------------------------------------------------------------------
app = FastAPI(title="Starbucks SRM AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# MEMORY
# -------------------------------------------------------------------------
memory_db: Dict[str, List[Dict[str, str]]] = {}

def save_memory(uid, role, msg):
    mem = memory_db.setdefault(uid, [])
    mem.append({"role": role, "msg": msg})
    if len(mem) > 10:
        mem.pop(0)

def get_memory(uid):
    return "\n".join([f"{m['role'].upper()}: {m['msg']}" for m in memory_db.get(uid, [])])

# -------------------------------------------------------------------------
# MASKING
# -------------------------------------------------------------------------
def mask(text):
    text = re.sub(r"\b\d{10}\b", "[PHONE_MASKED]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z-]+\.[A-Za-z]+", "[EMAIL_MASKED]", text)
    return text

# -------------------------------------------------------------------------
# RAG
# -------------------------------------------------------------------------
embed_model = None

def get_embed():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model

def chunk_text(txt, size=350, overlap=40):
    words = txt.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def build_rag(txt):
    chunks = chunk_text(txt)
    model = get_embed()
    emb = model.encode(chunks)
    emb = np.array(emb).astype("float32")
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(emb)
    faiss.write_index(index, INDEX_FILE)
    json.dump(chunks, open(CHUNKS_FILE, "w"))
    return len(chunks)

def rag_search(query, k=3):
    if not os.path.exists(INDEX_FILE):
        return []
    index = faiss.read_index(INDEX_FILE)
    chunks = json.load(open(CHUNKS_FILE))
    q_emb = get_embed().encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# -------------------------------------------------------------------------
# GEMINI
# -------------------------------------------------------------------------
def gemini(prompt):
    if not API_KEY:
        print("ERROR: No Gemini API key found.")
        return None
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content(prompt)
        return out.text
    except Exception as e:
        print("Gemini Error:", e)
        return None

# -------------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------------
@app.post("/api/upload_profile")
async def upload_profile(file: UploadFile = File(...)):
    name = file.filename.lower()
    if name.endswith(".pdf"):
        pdf = PdfReader(file.file)
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
    else:
        text = (await file.read()).decode()

    open(PROFILE_FILE, "w").write(text)
    count = build_rag(text)
    return {"indexed_chunks": count}

@app.post("/api/chat", response_model=ChatRes)
def chat(req: ChatReq):
    msg = mask(req.message)
    save_memory(req.user_id, "user", msg)

    rag = rag_search(msg)

    # distance
    try:
        dist = int(geodesic((req.lat, req.lon), (store["lat"], store["lon"])).meters)
    except:
        dist = 0

    inv = store_status["inventory"]
    is_open = store_status["open_now"]

    prompt = f"""
USER MESSAGE:
{msg}

STORE INFO:
Name: {store['name']}
Distance: {dist} meters
Offer: {store['offer']}
Open: {is_open}
Inventory: {inv}

RAG PROFILE:
{rag}

CONVERSATION MEMORY:
{get_memory(req.user_id)}

TASK:
Reply like a friendly Starbucks barista.
Use store details + inventory + profile context.
Max 2–3 sentences.
"""

    ans = gemini(prompt)

    if not ans:
        ans = f"Starbucks SRM Coffee Point is {dist}m away. {store['offer']}."

    save_memory(req.user_id, "assistant", ans)

    return ChatRes(
        reply=ans,
        personalized=True,
        features_used=["Gemini", "RAG", "Inventory", "Location"],
        store_info={
            "name": store["name"],
            "distance": dist,
            "offer": store["offer"],
            "open_now": is_open,
            "inventory": inv
        }
    )

# -------------------------------------------------------------------------
# NO UVICORN INSIDE FILE (avoids Windows issues)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Run the server using:")
    print("uvicorn app:app --host 0.0.0.0 --port 8000")
