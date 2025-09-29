import re
import unicodedata
from collections import Counter
from pathlib import Path
import io
import argparse
import os
import pymupdf
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytesseract import Output
from testing.ner_functions_init import ner_extraction, get_deadline, get_financial_details
import gen_ai1_init as gen_ai1

# ==================== CONFIGURATION ====================
MAX_CHUNK_TOKENS = 256
CHUNK_TOKEN_OVERLAP = 40
CLASSIFICATION_MODEL_NAME = "models/Fine_Tunned_Classi"  # Updated model path

BASE_DIR = Path(r"D:\clony\Doc_Load_Automation").resolve()
LOCAL_CLF_DIR = BASE_DIR / "models" / "classifier"

classification_dept_map = {
    0: "Finance",
    1: "Operations",
    2: "HR",
    3: "Engineering"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
NLP_MODEL = spacy.load("en_core_web_md")


def clean_text_english(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^A-Za-z0-9\s.,;:!?()'\-\"@%$&]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_page_text(page, doc):
    raw_text = ""
    for block in page.get_text("blocks"):
        txt = block[4].strip()
        if txt:
            raw_text += " " + txt

    images = page.get_images(full=True)
    for img in images:
        xref = img[0]
        try:
            img_data = doc.extract_image(xref)
            image = Image.open(io.BytesIO(img_data["image"]))
        except Exception:
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
        except Exception:
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


def load_classification_model():
    if LOCAL_CLF_DIR.exists():
        print(f"[INFO] Loading classification model from local cache: {LOCAL_CLF_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_CLF_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_CLF_DIR).to(device)
    else:
        print(f"[INFO] Downloading classification model from: {CLASSIFICATION_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_NAME).to(device)
        LOCAL_CLF_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_CLF_DIR)
        model.save_pretrained(LOCAL_CLF_DIR)
    return tokenizer, model


def classify_text_chunk(chunk, tokenizer, model):
    inputs = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).cpu().item()
    return classification_dept_map.get(pred, "Unknown")


def load_all_models():
    clf_tokenizer, clf_model = load_classification_model()
    nlp_model = NLP_MODEL
    return clf_tokenizer, clf_model, nlp_model


def pipeline_process_pdf(pdf_path, clf_tokenizer, clf_model, nlp_model):
    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = pymupdf.open(pdf_path)

    dept_votes = []
    deadlines_all = []
    financials_all = []

    for page_number, page in enumerate(doc, start=1):
        raw_text = extract_page_text(page, doc)
        if not raw_text:
            continue
        cleaned_text = clean_text_english(raw_text)
        if not cleaned_text:
            continue

        # ner_results = ner_extraction(cleaned_text, nlp_model)
        # deadlines_all.extend(ner_results.get("deadlines", []))
        # financials_all.extend(ner_results.get("financials", []))

        chunks = chunk_text_tokenwise(cleaned_text, tokenizer=clf_tokenizer)
        gen_ai1.encode(pdf_id, page_number,chunks)

        #for chunk in chunks:
            #dept = classify_text_chunk(chunk, clf_tokenizer, clf_model)
            #dept_votes.append(dept)

    #dominant_dept = Counter(dept_votes).most_common(1)[0][0] if dept_votes else "Unknown"

    summary = gen_ai1.create_summary(pdf_id)

    return {
        #"department": dominant_dept,
        "summary": summary,
        #"deadlines": deadlines_all,
        #"financials": financials_all,
    }



def highlight_text(pdf_path, terms, output_path="highlighted.pdf"):
    doc = pymupdf.open(pdf_path)

    for page_num, page in enumerate(doc):
        # ----- 1. Native text search highlighting -----
        for term in terms:
            text_instances = page.search_for(term)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

        # ----- 2. OCR highlighting for scanned PDFs -----
        # Render page as image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OCR with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        for i, word in enumerate(ocr_data['text']):
            for term in terms:
                if term.lower() in word.lower():
                    x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                                  ocr_data['width'][i], ocr_data['height'][i])
                    # Convert OCR coords to PDF coordinates
                    rect = pymupdf.Rect(x, y, x + w, y + h)
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()

    # Save PDF once after all highlights
    doc.save(output_path)
    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PDF Processing Pipeline")
    parser.add_argument("pdf_file", help="Path to input PDF file")
    args = parser.parse_args()

    print("[INFO] Loading all models...")
    tokenizer, model, nlp_model = load_all_models()

    print("[INFO] Processing PDF through pipeline...")
    results = pipeline_process_pdf(args.pdf_file, tokenizer, model, nlp_model)

    print("\n================ Pipeline Output ================\n")
    #print(f"Dominant Department: {results['department']}")
    print(f"\nSummary:\n{results['summary']}")
    # print(f"\nDeadlines found: {results['deadlines']}")
    # print(f"\nFinancial terms found: {results['financials']}")

    highlight_text(args.pdf_file, results['deadlines']+results['financials'])

    
