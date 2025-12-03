# 🚀 **One-Store Hyper-Personalized Customer Support Agent**

### *AI-powered assistant for Starbucks SRM Coffee Point — with RAG, location awareness, customer memory & privacy masking.*

---

## 🧠 1. **Problem Statement**

Retail customers expect **instant answers**:

* “Is this store open?”
* “Do you have Hot Cocoa right now?”
* “How far is the nearest branch?”
* “I’m cold — what should I get?”

Standard chatbots fail because they:
✔ Give generic answers
✔ Don’t know user context
✔ Don’t know store inventory
✔ Can’t personalize responses

---

## 🌟 2. **Our Solution**

We built a **Hyper-Personalized AI Agent** for a single coffee store:

> **User:** *“I’m cold.”*
> **AI:** *“You’re 40m away from Starbucks SRM Coffee Point. Hot Cocoa is in stock and you have a 10% coupon — come warm up!”*

This bot understands:

* Customer **memory** (chat history)
* Customer **profile RAG** (PDF/TXT ingestion)
* **Store inventory**
* **Store timing**
* Distance based on **user’s live GPS**
* Privacy-masked messages (phones/emails hidden)
* AI replies powered by **Gemini 1.5 Flash**

---

## 🏗️ 3. **System Architecture**

```
User → FastAPI → Privacy Layer → RAG Engine (FAISS)
     → Store Status → Location Distance
     → Gemini 1.5 Flash → Final Personalized Response
```

### **Core Features**

| Feature                        | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| 📌 **RAG on Customer Profile** | Upload PDF/TXT → chunks → embeddings → FAISS index |
| 📍 **Location Awareness**      | Computes distance between user GPS and store       |
| 🛍️ **Live Inventory Lookup**  | “Do you have Hot Cocoa?” → checks JSON inventory   |
| 🧠 **Conversation Memory**     | AI remembers last 10 messages                      |
| 🔒 **Privacy Masking**         | Removes phone numbers/emails before sending to LLM |
| 🤖 **Gemini 1.5 Flash**        | Final personalized response generator              |

---

## 📦 4. **Tech Stack**

| Layer          | Technology                   |
| -------------- | ---------------------------- |
| Backend        | FastAPI                      |
| AI Model       | Google Gemini 1.5 Flash      |
| RAG            | SentenceTransformers + FAISS |
| Embeddings     | all-MiniLM-L6-v2             |
| Location       | geopy                        |
| PDF Processing | PyPDF                        |
| Environment    | Python 3.11                  |
| Frontend       | HTML + Tailwind (optional)   |

---

## ⚙️ 5. **Setup Instructions**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/yourusername/hyper-personalized-agent.git
cd hyper-personalized-agent
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Add Your Gemini API Key**

Create `.env` file:

```
GEMINI_API_KEY=YOUR_KEY_HERE
```

### **4️⃣ Start Backend**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

You will see:

```
DEBUG: Loaded Gemini key → AIzaSy...
```

---

## 🧪 6. **How It Works (Step-by-Step)**

### 📝 Step 1 — Upload customer profile (optional)

Send a PDF or text file:

```
POST /api/upload_profile
```

RAG index is created automatically.

---

### 💬 Step 2 — Send chat messages

```
POST /api/chat
```

**Example input:**

```json
{
  "message": "I'm cold",
  "user_id": "cust01",
  "lat": 17.445,
  "lon": 78.391
}
```

**Bot response:**

> “You're 30m away from Starbucks SRM Coffee Point.
> Hot Cocoa is in stock and you have a 10% discount waiting.
> Come warm up inside!”

---

## 📍 7. **Store Configuration (data/store.json)**

```json
{
  "name": "Starbucks SRM Coffee Point",
  "lat": 17.445,
  "lon": 78.391,
  "offer": "10% off Hot Cocoa",
  "address": "SRM University AP Campus"
}
```

### 🔋 Inventory Example

(data/store_status.json)

```json
{
  "open_now": true,
  "inventory": {
    "Hot Cocoa": "Available",
    "Latte": "Low Stock",
    "Mocha": "Out of Stock"
  }
}
```

---

## 🤖 8. **AI Prompting Logic**

The system combines:

* User message
* Store info
* Distance
* Inventory
* Conversation memory
* RAG customer data

Then sends a structured prompt to **Gemini 1.5 Flash**, generating a **warm, barista-style** response.

---

## 🛡️ 9. **Privacy**

Before sending input to Gemini:

✔ Phone numbers → `[PHONE_MASKED]`
✔ Emails → `[EMAIL_MASKED]`
✔ No raw PII ever reaches LLM
✔ Compliant with hackathon privacy rules

---

## 🎯 10. **Why This Project Wins Hackathons**

* Real store problem
* Real customer experience improvement
* Uses RAG
* Uses location context
* Uses personalization
* Uses AI but also has **traditional ML components (FAISS)**
* Edge-case handling: open/close, inventory, distance
* Fully deployable

---

## 📸 11. **Screenshots / Demo (Add your UI images here)**

```

```

---

## ⭐ 12. **Future Enhancements**

* Multistore routing
* Real-time inventory sync
* Voice assistant support
* Loyalty points integration
* Multi-language AI responses

---

## 🏁 13. **Conclusion**

This project showcases how a **single coffee shop** can deliver an **enterprise-grade AI customer experience** with:

* Hyper personalization
* Ultra-fast retrieval
* Safety-first processing
* Real-world reliability
