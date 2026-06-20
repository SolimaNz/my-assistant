import os
import json
import hashlib
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
# File-type loaders -- maps extension to the right LangChain loader
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

    # Fallback -- try loading as plain text
    from langchain_community.document_loaders import TextLoader
    return TextLoader(filepath, encoding="utf-8")

# ---------------------------------------------------------------------------
# File hashing -- detect new/changed files
# ---------------------------------------------------------------------------
def _file_hash(filepath):
    """Compute SHA-256 hash of a file to detect changes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest(manifest_path):
    """Load the embedded files manifest (filename -> hash)."""
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest_path, manifest):
    """Save the embedded files manifest."""
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def setup_rag():
    data_dir = os.getenv("RAG_DATA_DIR", "data")
    chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
    manifest_path = os.path.join(chroma_dir, ".embedded_manifest.json")

    print(f"Starting RAG setup -- scanning '{data_dir}/' for files...")

    if not os.path.isdir(data_dir):
        print(f"Error: data directory '{data_dir}' not found!")
        return None

    # Ensure chroma directory exists (for the manifest file)
    os.makedirs(chroma_dir, exist_ok=True)

    # Load manifest of previously embedded files
    manifest = _load_manifest(manifest_path)
    new_manifest = {}

    # Categorize files: new, changed, unchanged, deleted
    files_to_embed = []   # (filename, filepath) -- need embedding
    files_unchanged = []  # filenames -- already embedded, skip

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        file_hash = _file_hash(filepath)
        new_manifest[filename] = file_hash

        if filename in manifest and manifest[filename] == file_hash:
            files_unchanged.append(filename)
        else:
            files_to_embed.append((filename, filepath))

    # Files that were embedded before but no longer exist in data/
    files_deleted = [f for f in manifest if f not in new_manifest]

    # Report status
    if files_unchanged:
        print(f"  [SKIP] {len(files_unchanged)} file(s) unchanged -- skipping")
    if files_deleted:
        print(f"  [DEL] {len(files_deleted)} file(s) removed from data/")

    if not files_to_embed and not files_deleted:
        print("No changes detected. Nothing to embed.")
        _save_manifest(manifest_path, new_manifest)
        embeddings = GeminiEmbeddings()
        db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        return db.as_retriever(search_kwargs={"k": 3})

    # Open existing ChromaDB
    embeddings = GeminiEmbeddings()
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    # Remove chunks for deleted or changed files
    files_to_purge = files_deleted + [f for f, _ in files_to_embed if f in manifest]
    for filename in files_to_purge:
        old_filepath = os.path.join(data_dir, filename)
        try:
            # LangChain loaders store the filepath in metadata "source"
            # Try both the current path and just the filename
            results = db.get(where={"source": old_filepath})
            if results and results["ids"]:
                db.delete(ids=results["ids"])
                print(f"  [DEL] Removed {len(results['ids'])} old chunks for {filename}")
        except Exception:
            pass  # file might not have been in DB yet

    # Embed new/changed files
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    total_chunks = 0

    for filename, filepath in files_to_embed:
        try:
            loader = _get_loader(filepath)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            if chunks:
                db.add_documents(chunks)
                total_chunks += len(chunks)
            status = "NEW" if filename not in manifest else "UPDATED"
            print(f"  [{status}] {filename} -- {len(chunks)} chunks embedded")
        except Exception as e:
            print(f"  [SKIP] Skipped {filename}: {e}")
            # Don't track failed files in manifest so they retry next run
            new_manifest.pop(filename, None)

    print(f"Done! Embedded {total_chunks} new chunks. API calls saved: skipped {len(files_unchanged)} unchanged file(s).")

    # Save updated manifest
    _save_manifest(manifest_path, new_manifest)

    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    retriever = setup_rag()