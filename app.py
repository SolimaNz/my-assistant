import os
import base64
import requests
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

embeddings = GeminiEmbeddings()

vector_store = Chroma(
    persist_directory=os.getenv("CHROMA_DIR", "chroma_db"),
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    question = data.get("question", "").strip()
    image_base64 = data.get("image", None)
    audio_base64 = data.get("audio", None)
    lat = data.get("lat", None)
    lng = data.get("lng", None)

    location_context = ""
    if lat and lng:
        location_context = f"The tourist is currently at coordinates: {lat}, {lng}."

    parts = []

    if audio_base64:
        audio_data = base64.b64decode(audio_base64)
        parts.append({"mime_type": "audio/mp3", "data": audio_data})
        if not question:
            question = "Please answer based on the audio."

    if image_base64:
        image_data = base64.b64decode(image_base64)
        parts.append({"mime_type": "image/jpeg", "data": image_data})
        if not question:
            question = "What is this? Provide tourist information about it."

    rag_context = ""
    if question:
        context_docs = retriever.invoke(question)
        rag_context = "\n\n".join([doc.page_content for doc in context_docs]).strip()

    prompt = f"""
You are an expert Egypt tourism assistant.
Help tourists with accurate, friendly, and detailed information.
Always respond in the same language the tourist used.

{location_context}

Context from knowledge base:
{rag_context}

Question: {question}

Answer:
"""
    parts.append(prompt)

    response = client.models.generate_content(
        model="models/gemini-3.1-flash-lite-preview",
        contents=parts
    )
    answer = response.text.strip()

    return jsonify({
        "source": "rag" if rag_context else "llm",
        "answer": answer
    })

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 5000)))