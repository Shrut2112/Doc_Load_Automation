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
    model_name=r"C:\Users\jains\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d",
    model_kwargs={"device": "cpu"}
)

def encode(pdf_id,page_numb,docs,encoder=encoder):
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
        "പ്രധാന സംഘടനാ പ്രവർത്തനങ്ങൾ, അടിയന്തരമായ നിർണായക ജോലികളും അവസാന തീയതികളും, അനുസരണവും നിയന്ത്രണാത്മകമായ പുതുക്കലുകളും, അന്തർ-വകുപ്പ് ഏകോപന പ്രശ്നങ്ങൾ, സ്റ്റാഫിംഗ്‌യും മാനവ വിഭവശേഷി മുൻഗണനകളും, സുരക്ഷാ ബുള്ളറ്റിനുകൾ, വാങ്ങൽ നില, അറിവ് സംരക്ഷണ വെല്ലുവിളികൾ, സമയബന്ധിതമായ തീരുമാനം കൈക്കൊള്ളലിനെയും പ്രവർത്തന കാര്യക്ഷമതയെയും ബാധിക്കുന്ന തന്ത്രപരമായ പ്രവർത്തനങ്ങൾ." 
        "സാമ്പത്തിക പ്രകടനം, ബജറ്റുകൾ, പേയ്‌മെന്റുകൾ, ഓഡിറ്റുകൾ, ചെലവ് നിയന്ത്രണം, ഫണ്ടിംഗ്, വാങ്ങൽ ധനകാര്യം."
    )

def query_pinecone_top_k(pdf_id, top_k=10,query=query):
    q_emb = encoder.embed_query(query)
    results = index.query(
    vector=q_emb,          # ✅ must use keyword
    top_k=10,
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
        all_results = index.query(
            vector=[0.0] * 384,   # dummy zero-vector
            top_k=top_k,
            include_metadata=True,
            filter={"pdf_id": pdf_id}
        )
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in all_results.get("matches", [])
        ]
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
        "6. If the text is in english print summary in english else print in hybrid malayalam and english\n"
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


def create_summary(pdf_id):
    docs = query_pinecone_top_k(pdf_id)

    try:
        summary = chain.invoke({"context": docs})
        print("Summary successfully generated.")
        # print("Summary:\n", summary.content if hasattr(summary, "content") else summary)
        return summary
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return f"Summary generation failed: {e}"
