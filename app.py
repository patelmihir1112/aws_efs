from fastapi import FastAPI
from langchain_community.embeddings import FakeEmbeddings  # Local embeddings, no API needed
from langchain_community.vectorstores import Chroma
import os

app = FastAPI()

# Path inside EFS
PERSIST_DIR = "/demo/chroma_db"

# Ensure directory exists
os.makedirs(PERSIST_DIR, exist_ok=True)

# Use FakeEmbeddings (generates random vectors, just for testing persistence)
embeddings = FakeEmbeddings(size=128)

# Initialize Chroma vectorstore
vstore = Chroma(
    collection_name="demo_collection",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)

@app.get("/")
def home():
    return {"message": "EFS + ECS + ChromaDB Demo is running without OpenAI!"}

@app.get("/add/{text}")
def add_text(text: str):
    """Add text to Chroma vectorstore (random embeddings)"""
    vstore.add_texts([text])
    vstore.persist()
    return {"status": "added", "text": text}

@app.get("/search/{query}")
def search_text(query: str):
    """Search similar text in Chroma vectorstore"""
    results = vstore.similarity_search(query, k=5)
    return {"query": query, "results": [r.page_content for r in results]}
