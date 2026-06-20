import os
import requests
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

# ---------------------------------------------------------------------------
# File-type loaders — maps extension to the right LangChain loader
# ---------------------------------------------------------------------------
def _get_loader(filepath):
    """Return the appropriate LangChain loader for the given file."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".txt", ".text"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(filepath, encoding="utf-8")

    if ext in (".md", ".markdown"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(filepath, encoding="utf-8")

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(filepath)

    if ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(filepath, encoding="utf-8")

    if ext == ".json":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(filepath, encoding="utf-8")

    if ext in (".docx", ".doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(filepath)

    if ext in (".html", ".htm"):
        try:
            from langchain_community.document_loaders import BSHTMLLoader
            return BSHTMLLoader(filepath, open_encoding="utf-8")
        except ImportError:
            from langchain_community.document_loaders import TextLoader
            return TextLoader(filepath, encoding="utf-8")

    # Fallback — try loading as plain text
    from langchain_community.document_loaders import TextLoader
    return TextLoader(filepath, encoding="utf-8")


def setup_rag():
    data_dir = os.getenv("RAG_DATA_DIR", "data")
    print(f"Starting RAG setup -- scanning '{data_dir}/' for files...")

    if not os.path.isdir(data_dir):
        print(f"Error: data directory '{data_dir}' not found!")
        return None

    # Collect documents from every file in the data folder
    all_docs = []
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue  # skip subdirectories

        try:
            loader = _get_loader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  [OK] Loaded {filename} ({len(docs)} document(s))")
        except Exception as e:
            print(f"  [SKIP] Skipped {filename}: {e}")

    if not all_docs:
        print("No documents loaded — nothing to embed.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = GeminiEmbeddings()
    print(f"Embedding {len(chunks)} chunks from {len(all_docs)} document(s) into ChromaDB...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=os.getenv("CHROMA_DIR", "chroma_db"))
    print("RAG setup complete!")
    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    retriever = setup_rag()