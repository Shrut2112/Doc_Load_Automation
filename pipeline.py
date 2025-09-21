# text_extract.py
import pymupdf 
from PIL import Image, ImageOps, ImageFilter
import io
import pytesseract
import spacy
from preprocess import clean_text_english, chunk_text
from ner_functions import get_deadline,get_financial_details,ner_extraction
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from classification_dept import classify
from collections import Counter

#loading tokenizer and fine tunned model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(r"D:\clony\Doc_Load_Automation\models\Fine_Tunned_Classi")

#loading ner model
nlp = spacy.load("en_core_web_md")

#extract text from the given page
def extract_page_text(page,doc,page_number):
        
        raw_text = ""

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
                    print(f"Page {page_number}: PDF text extracted.")
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
    
        return raw_text.strip()

def get_text_chunk(pdf):
    doc = pymupdf.open(pdf)
    #stores all deadlines and financial terms
    deadline = []
    fiancials = []
    #list for classified department chunk wise
    dept_list = []
    
    for page_number, page in enumerate(doc, start=1):
        #call function for text extraction
        raw_text = extract_page_text(page,doc,page_number)

        if not raw_text:
            print(f"Page {page_number}: No text found.")
            continue

        # 3) Clean English-only text
        #bsaic cleanning of text
        cleaned_text = clean_text_english(raw_text)
        if not cleaned_text:
            print(f"Page {page_number}: No English text found, skipping.")
            continue
        
        #extract deadline + financial terms from raw text
        ner_result = ner_extraction(cleaned_text,nlp)

        # Chunk text
        chunk = chunk_text(cleaned_text, max_length=1000, overlap=200)
        
        #classify department using fine tunned model for each chunk
        for chk in chunk:
            class_dept = classify(chk,tokenizer,model)
            dept_list.append(class_dept)

        #append deadlines + financial terms from each page in list 
        if ner_result['deadlines'] != [] or ner_result['financials'] != []:
            fiancials.extend(ner_result['financials'])
            deadline.extend(ner_result['deadlines'])
    
    #find frequency of department
    counter = Counter(dept_list)
    #most classified department
    most_common_dept = counter.most_common(1)[0][0]

    #json for the result
    results ={
        "deadlines":deadline,
        "financials":fiancials,
        "department":most_common_dept
    }
    return results

print(get_text_chunk("./pdfs/kochi_metro.pdf"))