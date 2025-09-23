import re
import unicodedata
from collections import Counter
from pathlib import Path
import io
import argparse

import pymupdf
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

# === Import your existing NER functions ===
from ner_functions import ner_extraction, get_deadline, get_financial_details

# ==================== CONFIG ====================
MAX_CHUNK_TOKENS = 800
CHUNK_TOKEN_OVERLAP = 100
CLASSIFICATION_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Local cache directories
BASE_DIR = Path(r"D:\clony\Doc_Load_Automation").resolve()
LOCAL_CLF_DIR = BASE_DIR / "models" / "classifier"
LOCAL_SUMM_DIR = BASE_DIR / "models" / "summarizer"

# Department mapping
classification_dept_map = {
    0: "HR",
    1: "Finance",
    2: "Operations",
    3: "Engineering"
}

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Spacy NLP model
NLP_MODEL = spacy.load("en_core_web_md")

# ==================== Utility Functions ====================
def clean_text_english(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^A-Za-z0-9\s.,;:!?()'\-\"@%$&]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_page_text(page, doc):
    raw_text = ""
    # Text blocks
    for block in page.get_text("blocks"):
        txt = block[4].strip()
        if txt:
            raw_text += " " + txt

    # Images + OCR
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            img_data = doc.extract_image(xref)
            image = Image.open(io.BytesIO(img_data["image"]))
        except:
            continue

        filtered = image.filter(ImageFilter.MedianFilter(size=3))
        gray = ImageOps.grayscale(filtered)
        scale = 300 / 72
        base_w = min(int(gray.width * scale), 2500)
        base_h = min(int(gray.height * scale), 2500)
        gray_resized = gray.resize((base_w, base_h), Image.LANCZOS)
        try:
            ocr_text = pytesseract.image_to_string(gray_resized)
            raw_text += " " + ocr_text
        except:
            continue
    return raw_text.strip()

def chunk_text_tokenwise(text, tokenizer, max_tokens=MAX_CHUNK_TOKENS, overlap=CHUNK_TOKEN_OVERLAP):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

# ==================== Model Loading (cache once) ====================
def load_classification_model():
    if LOCAL_CLF_DIR.exists():
        print(f"[INFO] Loading classification model from local cache: {LOCAL_CLF_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_CLF_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_CLF_DIR).to(device)
    else:
        print(f"[INFO] Downloading classification model: {CLASSIFICATION_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_NAME).to(device)
        LOCAL_CLF_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_CLF_DIR)
        model.save_pretrained(LOCAL_CLF_DIR)
    return tokenizer, model

def load_summarization_pipeline():
    if LOCAL_SUMM_DIR.exists():
        print(f"[INFO] Loading summarization pipeline from local cache: {LOCAL_SUMM_DIR}")
        pipe = hf_pipeline("summarization", model=str(LOCAL_SUMM_DIR), tokenizer=str(LOCAL_SUMM_DIR))
    else:
        print(f"[INFO] Downloading summarization model: {SUMMARIZATION_MODEL_NAME}")
        pipe = hf_pipeline("summarization", model=SUMMARIZATION_MODEL_NAME, tokenizer=SUMMARIZATION_MODEL_NAME)
        LOCAL_SUMM_DIR.mkdir(parents=True, exist_ok=True)
        pipe.model.save_pretrained(LOCAL_SUMM_DIR)
        pipe.tokenizer.save_pretrained(LOCAL_SUMM_DIR)
    return pipe

# ==================== Classification & Summarization ====================
def classify_text_chunk(chunk, tokenizer, model):
    inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).cpu().item()
    return classification_dept_map.get(pred, "Unknown")

def summarize_text_chunks(chunks, summarizer):
    chunk_summaries = []
    for ch in chunks:
        try:
            summary = summarizer(ch, max_length=150, min_length=40, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        except:
            chunk_summaries.append(ch[:150])
    fused_text = " ".join(chunk_summaries)
    try:
        final_summary = summarizer(fused_text, max_length=250, min_length=80, do_sample=False)[0]['summary_text']
    except:
        final_summary = fused_text[:250]
    return final_summary, chunk_summaries

# ==================== Main PDF Processing ====================
def process_pdf(pdf_path, clf_tokenizer, clf_model, summarizer):
    doc = pymupdf.open(pdf_path)
    dept_list, cleaned_chunks, deadlines, financials = [], [], [], []

    for page_number, page in enumerate(doc, start=1):
        raw_text = extract_page_text(page, doc)
        if not raw_text:
            continue
        cleaned_text = clean_text_english(raw_text)
        if not cleaned_text:
            continue

        # NER Extraction
        ner_results = ner_extraction(cleaned_text, NLP_MODEL)
        deadlines.extend(ner_results.get("deadlines", []))
        financials.extend(ner_results.get("financials", []))

        # Chunking + Classification
        chunks = chunk_text_tokenwise(cleaned_text, tokenizer=clf_tokenizer)
        for chk in chunks:
            dept = classify_text_chunk(chk, clf_tokenizer, clf_model)
            dept_list.append(dept)
            cleaned_chunks.append(chk)

    main_dept = Counter(dept_list).most_common(1)[0][0] if dept_list else "Unknown"
    final_summary, chunk_summaries = summarize_text_chunks(cleaned_chunks, summarizer)

    return {
        "department": main_dept,
        "summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "deadlines": deadlines,
        "financials": financials
    }

# ==================== Execute ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Processing Pipeline")
    parser.add_argument("pdf_file", help="Path to PDF file")
    args = parser.parse_args()

    print("[START] Loading models...")
    clf_tokenizer, clf_model = load_classification_model()
    summarizer = load_summarization_pipeline()

    print("[START] Processing PDF...")
    output = process_pdf(args.pdf_file, clf_tokenizer, clf_model, summarizer)

    print("\n================ OUTPUT ================\n")
    print("Dominant Department:", output["department"])
    print("\nSummary:\n", output["summary"])
    print("\nDeadlines found:", output["deadlines"])
    print("\nFinancial terms:", output["financials"])
