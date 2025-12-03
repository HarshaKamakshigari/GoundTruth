import re


PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s\-\(\)]{7,}\d)\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def mask_sensitive(text: str) -> str:
    """
    Masks phone numbers and email addresses.
    Names in your markdown are already fairly generic, so we leave them.
    """
    text = PHONE_PATTERN.sub("[PHONE_MASK]", text)
    text = EMAIL_PATTERN.sub("[EMAIL_MASK]", text)
    return text