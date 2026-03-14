import os
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"

    def _embed(self, text):
        response = requests.post(self.url, json={
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]}
        })
        data = response.json()
        return data["embedding"]["values"]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

def setup_rag():
    print("Starting RAG setup...")

    if os.path.exists("data/sample.txt"):
        loader = TextLoader("data/sample.txt", encoding="utf-8")
    else:
        print("Error: data/sample.txt not found!")
        return None

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = GeminiEmbeddings()

    print(f"Embedding {len(chunks)} chunks into ChromaDB...")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=os.getenv("CHROMA_DIR", "chroma_db")
    )

    print("RAG setup complete!")
    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    retriever = setup_rag()