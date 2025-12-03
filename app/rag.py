from .loaders import load_markdown
from .masking import mask_sensitive
from .vectorstore import VectorStore

class RAGPipeline:
    def __init__(self):
        self.vs = VectorStore()

    def load_data(self):
        store_entries = load_markdown("data/stores.md")
        user_entries  = load_markdown("data/users.md")

        all_texts = []
        for e in store_entries + user_entries:
            masked = mask_sensitive(e["text"])
            all_texts.append(masked)

        self.vs.add_texts(all_texts)

    def query(self, question):
        results = self.vs.search(question)
        context = "\n".join(results)
        return context
