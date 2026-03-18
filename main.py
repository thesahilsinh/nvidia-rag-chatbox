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

# ====================== CONFIG ======================
app = FastAPI(title="Sahilsinh RAG Chatbot")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is not set!")

# BEST MODEL FOR YOUR TASK (technical resume + malware/forensics blog)
llm = ChatNVIDIA(
    model="nvidia/llama-3.1-nemotron-70b-instruct",   # ← Best choice
    api_key=NVIDIA_API_KEY,
    temperature=0.6,
    max_tokens=1024
)

embeddings = NVIDIAEmbeddings(
    model="nvidia/embed-qa-4",
    api_key=NVIDIA_API_KEY
)

# ====================== LOAD DOCUMENTS ======================
print("Loading knowledge base...")

pdf_loader = DirectoryLoader("knowledge/", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
md_loader  = DirectoryLoader("knowledge/", glob="**/*.md",  loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)

docs = pdf_loader.load() + md_loader.load()
print(f"Loaded {len(docs)} documents from resume + blog posts")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

# ====================== VECTOR STORE ======================
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ====================== RAG PROMPT ======================
system_prompt = (
    "You are Sahilsinh Chavda's personal AI assistant. "
    "Answer ONLY using the provided context from his resume, Practical Malware Analysis labs, "
    "Drone Forensics Tool, teaching experience, and projects. "
    "Be professional, concise, and friendly. Cite specific parts when possible. "
    "If the answer is not in the context, say 'I don't have that information in my knowledge base.'\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ====================== API ======================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def health():
    return {"status": "alive", "model": "nvidia/llama-3.1-nemotron-70b-instruct"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = rag_chain.invoke({"input": request.message})
    return {"response": result["answer"]}

# ====================== START ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
