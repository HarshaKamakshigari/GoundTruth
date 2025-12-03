import os
import re
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from geopy.distance import geodesic

# -------------------------------------------------------
# LOAD ENV
# -------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("⚠️ No Gemini API Key — using fallback mode.")
else:
    print("Loaded Gemini Key:", api_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------------------------------
# LOAD STORE + USERS
# -------------------------------------------------------

store = {
    "name": "Starbucks SRM Coffee Point",
    "lat": 17.445,
    "lon": 78.391,
    "offer": "10% off Hot Cocoa"
}

users = {
    "CUST_001": {
        "name": "Alex Johnson",
        "favorite": "Hot Cocoa",
        "loyalty": "Gold Member"
    },
    "demo_user": {
        "name": "Demo User",
        "favorite": "Latte",
        "loyalty": "Standard"
    }
}

# -------------------------------------------------------
# FASTAPI
# -------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

# -------------------------------------------------------
# MODELS
# -------------------------------------------------------

class ChatReq(BaseModel):
    message: str
    user_id: str = "CUST_001"
    lat: float = 17.445
    lon: float = 78.391

class ChatRes(BaseModel):
    reply: str
    store_info: dict
    personalized: bool

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def mask(text):
    text = re.sub(r"\b\d{10}\b", "[PHONE_MASKED]", text)
    return text

def distance_m(lat, lon):
    return int(geodesic((lat, lon), (store["lat"], store["lon"])).meters)

def ai_reply(prompt):
    if not api_key:
        return None
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        print("Gemini error:", e)
        return None

# -------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------------

@app.post("/api/chat", response_model=ChatRes)
def chat(req: ChatReq):

    user_msg = mask(req.message)
    user = users.get(req.user_id, users["demo_user"])
    dist = distance_m(req.lat, req.lon)

    prompt = f"""
User Message: {user_msg}

User:
Name: {user['name']}
Favorite Drink: {user['favorite']}
Loyalty: {user['loyalty']}

Store:
Name: {store['name']}
Distance: {dist} meters
Offer: {store['offer']}

Write a warm, friendly reply in 2 sentences.
Mention user's name and favorite drink.
Mention distance and offer.
"""

    response = ai_reply(prompt)

    # fallback response
    if not response:
        response = (
            f"{user['name']}, you're {dist}m away from {store['name']}! "
            f"Come grab your favorite {user['favorite']} — today's offer is {store['offer']}."
        )

    return ChatRes(
        reply=response,
        personalized=True,
        store_info={
            "name": store["name"],
            "distance": dist,
            "offer": store["offer"]
        }
    )


# -------------------------------------------------------
# RUN MESSAGE
# -------------------------------------------------------
if __name__ == "__main__":
    print("Run using: uvicorn app:app --host 0.0.0.0 --port 8000")
