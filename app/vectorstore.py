from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def _init_(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(384)
        self.texts: List[str] = []
        self.metadatas: List[dict] = []

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        if not texts:
            return

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if len(self.texts) == 0:
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)

        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, dict]]:
        if len(self.texts) == 0:
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True)
        distances, ids = self.index.search(q_emb, min(k, len(self.texts)))

        results = []
        for idx in ids[0]:
            text = self.texts[idx]
            meta = self.metadatas[idx]
            results.append((text, meta))
        return results