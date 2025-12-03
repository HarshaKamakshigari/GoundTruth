import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add_texts(self, texts):
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query, k=3):
        q_emb = self.model.encode([query])
        distances, ids = self.index.search(np.array(q_emb), k)
        results = [self.texts[i] for i in ids[0]]
        return results
