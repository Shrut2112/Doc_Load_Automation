import os
import io
import uuid
import pymupdf
import chromadb
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from preprocess import clean_text_english, chunk_text  # your functions
from langchain.schema import Document

# ===================== CONFIG =====================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

persist_dir = "./chroma_db"
collection_name = "Doc_Embeddings"
pdf_path = "./pdfs/kochi_metro.pdf"
pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]  # e.g. "kochi_metro"

# Persistent Chroma
chroma_client = chromadb.PersistentClient(path=persist_dir)

try:
    collection = chroma_client.get_collection(name=collection_name)
except Exception:
    collection = chroma_client.create_collection(name=collection_name)

# Embeddings
encoder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # use HuggingFace shortcut
    model_kwargs={"device": "cpu"}
)

# ===================== FUNCTIONS =====================

def encode(encoder, docs):
    """Embed and store document chunks"""
    embeddings = encoder.embed_documents(docs)
    metadata = [{"pdf_id": pdf_id, "chunk_index": i} for i in range(len(docs))]
    ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadata,
        embeddings=embeddings
    )
    print(f"✅ Encoded {len(docs)} chunks and added to collection.")


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert organizational analyst. Your task is to generate a clear, concise, "
        "and professional summary of the given document. The summary must strictly use the "
        "provided context without adding hallucinations.\n\n"
        "Structure:\n"
        "1. Main Operations/Activities\n"
        "2. Challenges/Issues\n"
        "3. Key Points & Responsibilities\n"
        "4. Departmental Relevance\n"
    ),
    ("user", "Summarize the following document accordingly:\n\n{context}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=1000,
)

output_parser = StrOutputParser()
chain = create_stuff_documents_chain(llm, prompt=prompt, output_parser=output_parser)


def extract_page_text(page, doc, page_number):
    """Extract both PDF text and OCR from images"""
    raw_text = ""

    # Extract text blocks
    text_blocks = page.get_text("blocks")
    if text_blocks:
        for block in text_blocks:
            txt = block[4].strip()
            if txt:
                raw_text += " " + txt
        print(f"📄 Page {page_number}: PDF text extracted.")

    # Extract images and OCR
    images = page.get_images(full=True)
    if images:
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            try:
                img_data = doc.extract_image(xref)
                image = Image.open(io.BytesIO(img_data["image"]))
            except Exception as e:
                print(f"Page {page_number} Image {img_index}: extraction error {e}")
                continue

            gray = ImageOps.grayscale(image.filter(ImageFilter.MedianFilter(size=3)))
            scale = 300 / 72
            base_w = min(int(gray.width * scale), 2500)
            base_h = min(int(gray.height * scale), 2500)
            gray_resized = gray.resize((base_w, base_h), Image.LANCZOS)

            try:
                ocr_text = pytesseract.image_to_string(gray_resized)
                raw_text += " " + ocr_text
            except Exception as e:
                print(f"Page {page_number} OCR error: {e}")

    return raw_text.strip()


def create_summary(pdf_id, top_k=5, chain=chain):
    query = (
        "Main operations, challenges, financials, compliance, achievements, HR/staffing, "
        "departments, projects, deadlines, procurement, safety, strategic initiatives."
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=encoder,
        persist_directory=persist_dir,
    )

    docs = vectorstore.similarity_search(query=query, k=top_k, filter={'pdf_id': pdf_id})
    docs = vectorstore.similarity_search(query=query, k=top_k, filter={'pdf_id': pdf_id})
    
    if not docs:
        print("No relevant chunks found for the document.")
        return "No relevant chunks found for this document."
    
    # Wrap chunks in Document objects (required by stuff chain)
    doc_objects = [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]
    
    try:
        summary = chain.invoke({"context": doc_objects})
        print("Summary successfully generated.")
        print("Summary:\n", summary)  
        return summary
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return f"Summary generation failed: {e}"

def get_text_chunk(pdf):
    doc = pymupdf.open(pdf)

    for page_number, page in enumerate(doc, start=1):
        raw_text = extract_page_text(page, doc, page_number)
        if not raw_text:
            print(f"⚠️ Page {page_number}: No text found.")
            continue

        cleaned_text = clean_text_english(raw_text)
        if not cleaned_text:
            print(f"⚠️ Page {page_number}: No English text found.")
            continue

        chunks = chunk_text(cleaned_text, max_length=256, overlap=40)
        encode(encoder, chunks)

    return create_summary(pdf_id)


# ===================== MAIN =====================
if __name__ == "__main__":
    summary = get_text_chunk(pdf_path)
    print("\n📌 FINAL SUMMARY:\n", summary)
    #create_summary(pdf_id)
