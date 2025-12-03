from typing import List, Tuple, Optional
from .loaders import load_store_entries, load_user_entries
from .masking import mask_sensitive
from .vectorstore import VectorStore
import re


# Regex to extract IDs from markdown content
STORE_ID_RE = re.compile(r"\\*Store ID\\:\s([A-Z0-9]+)")
USER_ID_RE = re.compile(r"\\*User ID\\:\s([A-Z0-9]+)")


class RAGPipeline:
    def _init_(self):
        self.vs = VectorStore()
        self._data_loaded = False

        # lookup maps for exact store/user retrieval
        self.store_docs: dict[str, str] = {}
        self.user_docs: dict[str, str] = {}

    def load_data(self):
        """
        Loads stores and users from markdown, masks sensitive data,
        extracts IDs, builds the FAISS vector index.
        """
        if self._data_loaded:
            return

        store_entries = load_store_entries()
        user_entries = load_user_entries()

        texts: List[str] = []
        metadatas: List[dict] = []

        # Add store entries
        for entry in store_entries:
            raw_text = entry["text"]
            masked_text = mask_sensitive(raw_text)

            # Extract store ID
            m = STORE_ID_RE.search(raw_text)
            store_id = m.group(1) if m else None

            texts.append(masked_text)
            metadatas.append({
                "type": "store",
                "title": entry["title"],
                "store_id": store_id
            })

            if store_id:
                self.store_docs[store_id] = masked_text

        # Add user entries
        for entry in user_entries:
            raw_text = entry["text"]
            masked_text = mask_sensitive(raw_text)

            # Extract user ID
            m = USER_ID_RE.search(raw_text)
            user_id = m.group(1) if m else None

            texts.append(masked_text)
            metadatas.append({
                "type": "user",
                "title": entry["title"],
                "user_id": user_id
            })

            if user_id:
                self.user_docs[user_id] = masked_text

        # Add all text to FAISS
        self.vs.add_texts(texts, metadatas)
        self._data_loaded = True

    def retrieve(self, question: str, k: int = 5) -> List[Tuple[str, dict]]:
        return self.vs.search(question, k=k)

    def build_context(
        self,
        question: str,
        user_id: Optional[str] = None,
        store_id: Optional[str] = None,
        k: int = 5
    ) -> str:
        """
        Builds context including:
        - The specific user (if provided)
        - The specific store (if provided)
        - The top-k semantic matches from FAISS
        """
        parts: List[str] = []

        # Include exact user info
        if user_id and user_id in self.user_docs:
            parts.append(f"[USER: {user_id}]\n{self.user_docs[user_id]}")

        # Include exact store info
        if store_id and store_id in self.store_docs:
            parts.append(f"[STORE: {store_id}]\n{self.store_docs[store_id]}")

        # Semantic matches
        results = self.retrieve(question, k=k)

        for text, meta in results:
            # Avoid duplicates
            if meta.get("user_id") == user_id:
                continue
            if meta.get("store_id") == store_id:
                continue

            label = f"{meta['type'].upper()} - {meta['title']}"
            parts.append(f"[{label}]\n{text}")

        return "\n\n---\n\n".join(parts)