import os
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress annoying Pydantic v1 warning on Python 3.14 / 3.12
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
    category=UserWarning,
    module="langchain_core"
)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# NEW CORRECT IMPORTS FOR LANGCHAIN 0.3+
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

app = FastAPI(title="Sahilsinh RAG Chatbot")

# ====================== LAZY INIT ======================
llm = None
retriever = None
rag_chain = None

def initialize_rag():
    global llm, retriever, rag_chain
    if rag_chain is not None:
        return

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is missing!")

    print("🔑 NVIDIA API Key loaded")

    # DeepSeek-V3.2 (best available on NVIDIA right now)
    llm = ChatNVIDIA(
        model="deepseek-ai/deepseek-v3_2",
        api_key=NVIDIA_API_KEY,
        temperature=0.6,
        max_tokens=1024
    )

    embeddings = NVIDIAEmbeddings(
        model="nvidia/embed-qa-4",
        api_key=NVIDIA_API_KEY
    )

    print("📂 Loading knowledge base...")
    try:
        pdf_loader = DirectoryLoader("knowledge/", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        md_loader  = DirectoryLoader("knowledge/", glob="**/*.md",  loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
        docs = pdf_loader.load() + md_loader.load()
        print(f"✅ Loaded {len(docs)} documents")
    except Exception as e:
        print(f"⚠️ Document warning: {e}")
        docs = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    system_prompt = (
        "You are Sahilsinh Chavda's personal AI assistant. "
        "Answer ONLY using the provided context from his resume, PMA labs, "
        "Drone Forensics Tool, teaching experience, and projects. "
        "Be professional, concise, and friendly.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("🚀 RAG initialized successfully with DeepSeek-V3.2!")

# ====================== ENDPOINTS ======================
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def health():
    return {"status": "alive", "model": "deepseek-ai/deepseek-v3_2"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    initialize_rag()
    result = rag_chain.invoke({"input": request.message})
    return {"response": result["answer"]}

# ====================== START ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
