import re
from typing import List, Dict


def _split_sections(text: str, marker: str) -> List[Dict]:
    """
    Split the markdown by a marker like '### Store' or '### User',
    regardless of whether it's on a new line or in the middle of a line.
    """
    # Ensure consistent newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Add a newline before every marker so splitting is easier
    # e.g. "## ... ### Store 1" -> "## ... \n### Store 1"
    text = text.replace(marker, f"\n{marker}")

    # Now split on the marker, keeping the marker via lookahead
    parts = re.split(rf"(?={marker})", text)

    sections: List[Dict] = []
    for part in parts:
        part = part.strip()
        if not part.startswith(marker):
            continue

        lines = part.splitlines()
        title_line = lines[0]               # e.g. "### Store 1"
        title = title_line.lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip() # rest of the block

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