from pinecone import Pinecone
import os
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
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
from pinecone import ServerlessSpec
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "doc-embeddings"
if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384,metric="cosine",
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
         )
        )

index = pc.Index(index_name)

encoder = HuggingFaceEmbeddings(
    model_name=r"C:\Users\jains\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    model_kwargs={"device": "cpu"}
)

def encode(pdf_id,encoder,page_numb,docs):
    """Embed and store document chunks"""
    embeddings = encoder.embed_documents(docs)
    vectors = [
        (str(f"{pdf_id}_{page_numb}_{i}"), emb, {"pdf_id": pdf_id, "chunk_index": i,"page_no":page_numb,"text":docs[i]})
        for i, emb in enumerate(embeddings)
    ]
    index.upsert(vectors)
    print(f"✅ {len(vectors)} chunks stored in Pinecone.")

query = (
        "Key organizational operations, critical urgent tasks and deadlines, compliance and regulatory updates, "
        "inter-departmental coordination issues, staffing and HR priorities, safety bulletins, procurement status, "
        "knowledge retention challenges, and strategic initiatives impacting timely decision-making and operational efficiency."
        "financial performance, budgets, payments, audits, cost control, funding, procurement finance"
    )

def query_pinecone_top_k(query,pdf_id, top_k=10):
    q_emb = encoder.embed_query(query)
    results = index.query(
    vector=q_emb,          # ✅ must use keyword
    top_k=top_k,
    include_metadata=True,
    filter={"pdf_id": pdf_id}
    )
    docs = [
        Document(
            page_content=match['metadata'].get('text', ''),
            metadata=match['metadata']
        )
        for match in results['matches']
    ]
    if not docs:
        print("No doc found")
    else:
        print("Chunks found: ",len(docs))
        
    return docs

prompt = ChatPromptTemplate.from_messages([
     (
        "system",
        "You are an expert organizational analyst. Generate a brief, actionable summary that highlights the most important and urgent points "
        "from the given document chunks. The summary should focus on tasks department heads need to act on immediately, critical deadlines, compliance, "
        "and cross-department coordination issues. Use only the provided context strictly.\n\n"
        "Structure:\n"
        "1. Overview of Main Operations and Activities\n"
        "2. Critical Urgent Tasks and Immediate Deadlines\n"
        "3. Compliance and Regulatory Highlights\n"
        "4. Key Departmental Responsibilities and Coordination Needs\n"
        "5. Safety, Staffing, Procurement, and Strategic Initiatives\n"
    ),
    ("user", "Summarize the following document accordingly:\n\n{context}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=1500,
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
        encode(pdf_id,encoder, page_number,chunks)

    summary = create_summary(pdf_id)
    return summary

def create_summary(pdf_id):
    docs = query_pinecone_top_k(query,pdf_id)

    try:
        summary = chain.invoke({"context": docs})
        print("Summary successfully generated.")
        # print("Summary:\n", summary.content if hasattr(summary, "content") else summary)
        return summary
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return f"Summary generation failed: {e}"

# ===================== MAIN =====================
if __name__ == "__main__":
    pdf_path = "./pdfs/kochi_metro.pdf"
    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]  # e.g. "kochi_metro"

    #summary = create_summary(pdf_id)
    summary = get_text_chunk(pdf_path)
    print("\n📌 FINAL SUMMARY:\n", summary)
    