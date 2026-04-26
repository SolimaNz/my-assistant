__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import base64
import requests
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------
# System instruction — gives the model its identity and personality
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are Soli, a passionate and deeply knowledgeable Egypt tourism guide.
You have encyclopedic knowledge of Egyptian history (ancient, Islamic, Coptic, and modern),
landmarks, culture, cuisine, travel logistics, safety, local etiquette, and hidden gems.

Your personality:
- Warm, enthusiastic, and conversational — you genuinely love Egypt and it shows
- Share insider tips and fascinating stories that typical guidebooks miss
- Paint vivid pictures with your words so the tourist can imagine the experience
- Be concise for simple questions, richly detailed when the topic deserves depth
- Always respond in the exact language the tourist uses (Arabic, English, French, etc.)
- When shown an image or video, identify it confidently and give rich tourist context
- When given GPS coordinates, tailor everything specifically to what is nearby
- Never give generic, robotic bullet-point lists — speak like a knowledgeable friend
- If the tourist seems lost or confused, gently guide them and offer next steps"""

# ---------------------------------------------------------------------------
# Embeddings + ChromaDB
# ---------------------------------------------------------------------------
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
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ---------------------------------------------------------------------------
# /ask endpoint
# ---------------------------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    question      = data.get("question", "").strip()
    image_base64  = data.get("image", None)       # base64 JPEG (backward compat)
    audio_base64  = data.get("audio", None)       # base64 MP3  (backward compat)
    video_base64  = data.get("video", None)       # base64 MP4
    files         = data.get("files", [])         # [{"mime_type": "...", "data": "<base64>"}]
    lat           = data.get("lat", None)
    lng           = data.get("lng", None)
    history       = data.get("history", [])       # [{"role": "user"/"model", "content": "..."}]

    # --- Location context ---------------------------------------------------
    location_context = ""
    if lat and lng:
        location_context = (
            f"The tourist is currently at GPS coordinates: "
            f"latitude {lat}, longitude {lng}. "
            f"Tailor all suggestions specifically to this exact location."
        )

    # --- Media parts --------------------------------------------------------
    media_parts = []
    media_labels = []

    if image_base64:
        media_parts.append(
            types.Part.from_bytes(data=base64.b64decode(image_base64), mime_type="image/jpeg")
        )
        media_labels.append("an image")
        if not question:
            question = "What is this landmark or place? Give me rich tourist information about it."

    if audio_base64:
        media_parts.append(
            types.Part.from_bytes(data=base64.b64decode(audio_base64), mime_type="audio/mp3")
        )
        media_labels.append("an audio clip")
        if not question:
            question = "Please listen to this audio and respond as an Egypt tourism guide."

    if video_base64:
        media_parts.append(
            types.Part.from_bytes(data=base64.b64decode(video_base64), mime_type="video/mp4")
        )
        media_labels.append("a video")
        if not question:
            question = "What can you tell me about this video from a tourist perspective?"

    # Generic files — any MIME type the model supports
    # (PDF, DOCX, PNG, WAV, MP4, TXT, CSV, …)
    for f in files:
        mime = f.get("mime_type", "application/octet-stream")
        raw  = base64.b64decode(f.get("data", ""))
        media_parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
        media_labels.append(mime)
        if not question:
            ext = mime.split("/")[-1]
            question = f"Please analyze this {ext} file and provide relevant Egypt tourism information."

    # --- RAG retrieval ------------------------------------------------------
    rag_context = ""
    if question:
        context_docs = retriever.invoke(question)
        rag_context = "\n\n".join([doc.page_content for doc in context_docs]).strip()

    # --- Build conversation history text ------------------------------------
    history_text = ""
    if history:
        lines = []
        for turn in history:
            role = "Tourist" if turn.get("role") == "user" else "Soli"
            lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n".join(lines)

    # --- Compose prompt -----------------------------------------------------
    sections = []

    if location_context:
        sections.append(f"[LOCATION]\n{location_context}")

    if rag_context:
        sections.append(f"[KNOWLEDGE BASE]\n{rag_context}")

    if history_text:
        sections.append(f"[CONVERSATION SO FAR]\n{history_text}")

    if media_labels:
        sections.append(f"[MEDIA] The tourist has shared: {', '.join(media_labels)}")

    sections.append(f"Tourist: {question}\nSoli:")

    prompt = "\n\n".join(sections)

    # --- Call Gemini --------------------------------------------------------
    contents = media_parts + [prompt]

    response = client.models.generate_content(
        model="models/gemini-3.1-flash-lite-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.75,
            top_p=0.95,
            max_output_tokens=2048,
        )
    )

    return jsonify({"answer": response.text.strip()})

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 5000)))