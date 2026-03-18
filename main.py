import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI(title="Sahilsinh RAG Chatbot")

# ====================== LAZY INIT ======================
llm = None
retriever = None
rag_chain = None

def initialize_rag():
    global llm, retriever, rag_chain
    if rag_chain is not None:
        return  # already initialized

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is missing in Render!")

    print("🔑 Using NVIDIA API Key (starts with):", NVIDIA_API_KEY[:10] + "...")

    # BEST MODEL FOR RESUME + MALWARE ANALYSIS
    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0.6,
        max_tokens=1024
    )

    embeddings = NVIDIAEmbeddings(
        model="nvidia/embed-qa-4",
        api_key=NVIDIA_API_KEY
    )

    # Load documents (safe if folder is empty)
    print("📂 Loading knowledge base...")
    try:
        pdf_loader = DirectoryLoader("knowledge/", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        md_loader  = DirectoryLoader("knowledge/", glob="**/*.md",  loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
        docs = pdf_loader.load() + md_loader.load()
        print(f"✅ Loaded {len(docs)} documents")
    except Exception as e:
        print(f"⚠️ Document loading warning: {e}. Using empty knowledge base for now.")
        docs = []

    if not docs:
        print("⚠️ No documents found! Add Sahilsinh_chavda_resume.pdf to knowledge/ folder.")

    # Split & vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # RAG chain
    system_prompt = (
        "You are Sahilsinh Chavda's personal AI assistant. "
        "Answer ONLY using the provided context from his resume, PMA labs, Drone Forensics Tool, "
        "teaching experience, and projects. Be professional, concise, and friendly.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("🚀 RAG system initialized successfully!")

# ====================== ENDPOINTS ======================
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def health():
    return {"status": "alive", "model": "nvidia/llama-3.1-nemotron-70b-instruct"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    initialize_rag()   # ← only runs once, safely

    result = rag_chain.invoke({"input": request.message})
    return {"response": result["answer"]}

# ====================== START ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
