from langchain_rag.embedding import EmbedFromS3
# app/main.py
from fastapi import FastAPI

app = FastAPI(title="Tranner RAG API", version="1.0.0")

@app.get("/")
def root():
    return {"msg": "Hello FastAPI!"}
