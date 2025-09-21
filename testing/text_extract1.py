# text_extract.py
import pymupdf 
from PIL import Image, ImageOps, ImageFilter
import io
import pytesseract
import sys
import os
import csv

from preprocess import clean_text_english, chunk_text

# Optional: force UTF-8 stdout (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Set Tesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_FILE = "pdfs/kochi_metro.pdf"
OUTPUT_CSV = "extracted.csv"

def append_chunks_to_csv(chunks, output_csv, filename, page_number=None):
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source_file','page','chunk_index','text'])
        if not file_exists:
            writer.writeheader()
        for idx, chunk in enumerate(chunks):
            writer.writerow({
                'source_file': filename,
                'page': page_number,
                'chunk_index': idx,
                'text': chunk
            })

print(f"Opening PDF: {PDF_FILE}")
doc = pymupdf.open(PDF_FILE)

for page_number, page in enumerate(doc, start=1):
    raw_text = ""
    # 1) Extract PDF text layer
    text_blocks = page.get_text("blocks")
    if text_blocks:
        for block in text_blocks:
            txt = block[4].strip()
            if txt:
                raw_text += " " + txt
        print(f"Page {page_number}: PDF text extracted.")

    # 2) Extract images & OCR
    images = page.get_images(full=True)
    if images:
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            try:
                img_data = doc.extract_image(xref)
                image_bytes = img_data["image"]
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                print(f"Page {page_number} Image {img_index}: extraction/open error: {e}")
                continue

            # Preprocess image for OCR
            filtered = image.filter(ImageFilter.MedianFilter(size=3))
            gray = ImageOps.grayscale(filtered)
            scale = 300 / 72
            base_w = min(int(gray.width * scale), 2500)
            base_h = min(int(gray.height * scale), 2500)
            gray_resized = gray.resize((base_w, base_h), Image.LANCZOS)

            # OCR
            try:
                ocr_text = pytesseract.image_to_string(gray_resized)
                raw_text += " " + ocr_text
            except Exception as e:
                print(f"Page {page_number} Image {img_index}: OCR error: {e}")
                continue

    # 3) Clean English-only text
    cleaned_text = clean_text_english(raw_text)
    if not cleaned_text:
        print(f"Page {page_number}: No English text found, skipping.")
        continue

    # 4) Chunk text
    chunks = chunk_text(cleaned_text, max_length=1000, overlap=200)

    # 5) Append chunks to CSV
    append_chunks_to_csv(chunks, OUTPUT_CSV, filename=os.path.basename(PDF_FILE), page_number=page_number)
    print(f"Page {page_number}: {len(chunks)} chunks written to CSV.")

print(f"All done. Output saved to {OUTPUT_CSV}")
