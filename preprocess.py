# preprocess.py
import re
import unicodedata

def clean_text_english(text):
    """
    1) Normalize unicode
    2) Keep only English letters, digits, punctuation, basic symbols
    3) Remove extra spaces / newlines
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Remove non-English characters (keep basic punctuation and digits)
    text = re.sub(r"[^A-Za-z0-9\s.,;:!?()'\-\"@%$&]", " ", text)
    
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def chunk_text(text, max_length=1000, overlap=200):
    """
    Chunk text into overlapping windows
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks
