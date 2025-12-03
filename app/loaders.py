import markdown
import re

def load_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("# ")[1:]  # split by entries
    entries = []

    for block in blocks:
        lines = block.strip().split("\n")
        title = lines[0]
        body = "\n".join(lines[1:])
        entries.append({
            "title": title,
            "text": body
        })

    return entries
