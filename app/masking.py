import re

def mask_sensitive(text):
    text = re.sub(r"\b\d{10}\b", "[PHONE_MASK]", text)
    text = re.sub(r"[A-Za-z]+ [A-Za-z]+", "[NAME_MASK]", text)
    return text
