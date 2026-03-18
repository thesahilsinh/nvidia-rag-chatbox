from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Your NVIDIA key
llm = ChatNVIDIA(api_key="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", model="deepseek-ai/deepseek-v3.2")
embeddings = NVIDIAEmbeddings(api_key="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")

# Auto-load resume + blog
loader = DirectoryLoader("knowledge/", glob="**/*.pdf") + DirectoryLoader("knowledge/", glob="**/*.md")
docs = loader.load()

vectorstore = Chroma.from_documents(docs, embeddings)
# Then simple RAG chain + FastAPI endpoint /chat

# client = ChatNVIDIA(
#   model="deepseek-ai/deepseek-v3.2",
#   api_key="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", 
#   temperature=1,
#   top_p=0.95,
#   max_tokens=8192,
#   extra_body={"chat_template_kwargs": {"thinking":False}},
# )

# for chunk in client.stream([{"role":"user","content":""}]):
  
#     if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
#       print(chunk.additional_kwargs["reasoning_content"], end="")
  
#     print(chunk.content, end="")

