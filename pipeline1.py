import re
import unicodedata
from pathlib import Path
import io

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from collections import Counter

# === CONFIG ===
MAX_CHUNK_TOKENS = 800
CHUNK_TOKEN_OVERLAP = 100
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
NLP_MODEL = spacy.load("en_core_web_md")
device = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_CLF_DIR = Path(r"D:\clony\Doc_Load_Automation\models\Fine_Tunned_Classi").resolve()
clf_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
clf_model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_CLF_DIR, local_files_only=True
).to(device)
summ_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL).to(device)
from ner_functions import ner_extraction  # Must be thread-safe
from classification_dept import classify  # Should support vectorized inference

def clean_text_english(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^A-Za-z0-9\s.,;:!?()'\-\"@%$&]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_page_text_simple(page):
    # Try digital text first
    raw_text = ""
    for block in page.get_text("blocks"):
        txt = block[4].strip()
        if txt:
            raw_text += " " + txt
    # OCR fallback if no text
    if not raw_text.strip():
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            try:
                img_data = page.parent.extract_image(xref)
                image = Image.open(io.BytesIO(img_data["image"]))
                filtered = image.filter(ImageFilter.MedianFilter(size=3))
                gray = ImageOps.grayscale(filtered)
                scale = 300 / 72
                base_w = min(int(gray.width * scale), 2500)
                base_h = min(int(gray.height * scale), 2500)
                gray_resized = gray.resize((base_w, base_h), Image.LANCZOS)
                ocr_text = pytesseract.image_to_string(gray_resized, lang="eng")
                raw_text += " " + ocr_text
            except Exception:
                continue
    return raw_text.strip()

def chunk_text_tokenwise(text, max_tokens=MAX_CHUNK_TOKENS, overlap=CHUNK_TOKEN_OVERLAP):
    token_ids = summ_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = summ_tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

def batch_classify_text(chunks):
    inputs = clf_tokenizer(chunks, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
    return preds

def batch_summarize_chunks(chunks):
    all_summaries = []
    batch_size = 8
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = summ_tokenizer(batch, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
        with torch.no_grad():
            summary_ids = summ_model.generate(
                inputs.input_ids,
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summaries = [summ_tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summary_ids]
        all_summaries.extend(summaries)
    fused_text = " ".join(all_summaries)
    inputs = summ_tokenizer([fused_text], return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
    with torch.no_grad():
        final_summary_ids = summ_model.generate(
            inputs.input_ids,
            max_length=250,
            min_length=80,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    final_summary = summ_tokenizer.decode(final_summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return final_summary, all_summaries

def process_pdf_with_summary(pdf_path):
    doc = fitz.open(pdf_path)
    dept_list, deadlines, financials, cleaned_chunks = [], [], [], []
    # --- Serial, but fast: open each page, extract text OCR only if necessary.
    for page_number in range(doc.page_count):
        page = doc[page_number]
        raw_text = extract_page_text_simple(page)
        if not raw_text:
            continue
        cleaned_text = clean_text_english(raw_text)
        if not cleaned_text:
            continue
        ner_results = ner_extraction(cleaned_text, NLP_MODEL)
        if ner_results['deadlines']:
            deadlines.extend(ner_results['deadlines'])
        if ner_results['financials']:
            financials.extend(ner_results['financials'])
        chunks = chunk_text_tokenwise(cleaned_text)
        cleaned_chunks.extend(chunks)
    # Batch classification
    if cleaned_chunks:
        dept_preds = batch_classify_text(cleaned_chunks)
        dept_list.extend(dept_preds)
    department_count = Counter(dept_list)
    main_department = department_count.most_common(1)[0][0] if dept_list else None
    # Batch summarization
    if cleaned_chunks:
        final_summary, chunk_summaries = batch_summarize_chunks(cleaned_chunks)
    else:
        final_summary, chunk_summaries = "", []
    return {
        "deadlines": deadlines,
        "financials": financials,
        "department": main_department,
        "summary": final_summary,
        "chunk_summaries": chunk_summaries
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FAST PDF processing: NER, Classification, Summarization")
    parser.add_argument("pdf_file", help="Path to input PDF")
    args = parser.parse_args()
    output = process_pdf_with_summary(args.pdf_file)
    print("Summary:\n", output["summary"])
    print("\nDeadlines found:", output["deadlines"])
    print("\nFinancial terms:", output["financials"])
    print("\nDominant Department:", output["department"])
