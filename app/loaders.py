import re
from typing import List, Dict


def _split_sections(text: str, marker: str) -> List[Dict]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(marker, f"\n{marker}")
    parts = re.split(rf"(?={marker})", text)

    sections: List[Dict] = []
    for part in parts:
        part = part.strip()
        if not part.startswith(marker):
            continue

        lines = part.splitlines()
        title_line = lines[0]                   
        title = title_line.lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()     

        sections.append({"title": title, "text": body})

    return sections


def load_store_entries(path: str = "data/stores.md") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return _split_sections(text, "### Store")


def load_user_entries(path: str = "data/users.md") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return _split_sections(text, "### User")