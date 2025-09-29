from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gen_ai1_init import query_pinecone_top_k

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=1500,
)

def qna(pdf_id,query):
    docs = query_pinecone_top_k(pdf_id,query)
    if not docs:
        return "No relevant information found."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question. Don't make up answers. Don't hallucinate."),
        ("system", "If the context does not contain the answer, respond with 'No relevant information found. Give output in same language as input question.'"),
        ("system", "Context:\n{context}"),
        ("user", "Question: {question}"),
    ])

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context",
        output_parser=StrOutputParser(),
    )

    response = chain.invoke({"context":docs, "question":query})
    return response

pdf_id = "finance"
query = "ഈ പ്രമാണത്തിലെ പ്രധാനപ്പെട്ട വിഷയം എന്താണ്?"
answer = qna(pdf_id, query)
print("Answer:", answer)