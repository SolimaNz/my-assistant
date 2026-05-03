__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import base64
import requests
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------
# Model fallback — try primary first, then backup if overloaded
# ---------------------------------------------------------------------------
MODELS = [
    "models/gemini-3.1-flash-lite-preview",  # primary — 15 RPM, 500 RPD
    "models/gemini-2.5-flash-lite-preview",  # backup  — 10 RPM
]

def call_gemini(contents, config):
    """Try each model in order. Return response or raise on total failure."""
    last_error = None
    for model in MODELS:
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            print(f"[WARN] {model} failed: {e}. Trying next model...")
            continue
    raise last_error

# ---------------------------------------------------------------------------
# Live Search — DuckDuckGo (Free, Unlimited)
# ---------------------------------------------------------------------------
def get_live_context(query):
    """Searches DuckDuckGo and returns top 3 results as context."""
    try:
        results = DDGS().text(query, max_results=3)
        if results:
            context = "Live search results:\n"
            for r in results:
                context += f"- {r.get('title')}: {r.get('body')}\n"
            return context
    except Exception as e:
        print(f"[WARN] Live search failed: {e}")
    return ""

# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------
CHAT_SYSTEM = """You are Soli, a smart and friendly virtual assistant for anyone in Egypt.
You're not just a tour guide — you're a local expert who helps with EVERYTHING:
tourism, daily life, emergencies, language, transportation, shopping, food,
health, legal questions, scams, cultural tips, and anything else someone
in Egypt might need help with.

Core rules:
- Match the user's energy. Short question → short answer. Deep question → deep answer.
  "Where's the nearest ATM?" gets one sentence, not a history lesson.
- Talk like a real person, not a textbook. Use natural, flowing language.
- Always reply in the same language the user uses.
- When shown media (photos, videos, audio, files), identify what you see/hear
  and give useful context about it.
- When GPS coordinates are provided, be hyper-specific to that exact location.
- Share insider tips, local slang, price expectations, and safety heads-ups
  when they're relevant — don't force them into every answer.
- Help with practical stuff: SIM cards, currency exchange, tipping culture,
  taxi fares, pharmacy locations, embassy contacts, bargaining advice,
  common scams, Arabic phrases, restaurant recommendations, and more.
- If someone is in trouble (lost passport, medical issue, police),
  give clear and calm step-by-step guidance.
- Go beyond typical tourist spots. Recommend the real local Egyptian experience:
  local eateries (محلات الفول والطعمية والكشري), قهاوي بلدي, أسواق شعبية,
  street food carts, local neighborhoods, parks where families hang out,
  popular local restaurants that tourists would never find on their own.
  The goal is to let the user live like an Egyptian, not just visit like a tourist.
- If you don't know something, say so honestly rather than making it up.
- Never start your response with "Great question!" or similar filler."""

PLAN_SYSTEM = """You are an expert Egypt trip planner.
Generate a detailed day-by-day travel itinerary based on the user's request.
Return ONLY valid JSON matching the exact schema below — no extra text, no markdown.

JSON Schema:
{
  "title": "string — a creative name for the trip",
  "days": [
    {
      "total_cost_usd": number,
      "total_cost_egp": number,
      "activities": [
        {
          "time": "string — e.g. 09:00 AM",
          "title": "string — place or activity name",
          "description": "string — one engaging sentence about what to do there",
          "cost_usd": number,
          "cost_egp": number,
          "lat": number — latitude of this place,
          "lng": number — longitude of this place,
          "category": "string — one of: attraction, food, shopping, experience, transport"
        }
      ]
    }
  ]
}

Rules:
- Each day should have 3-5 activities with realistic timing
- Include a mix of categories: attractions, food, experiences
- Costs should be realistic for Egypt in both USD and EGP
- lat and lng must be accurate real-world coordinates for each place
- Descriptions should be vivid and enticing, not generic
- If the user specifies a budget, keep total costs within it
- If the user specifies interests, prioritize those categories
- Order activities logically by geography to minimize travel time
- Don't only suggest tourist landmarks. Mix in authentic local spots that
  regular Egyptians actually go to: popular local restaurants, street food,
  neighborhood cafes, local markets, parks, and everyday cultural experiences.
  The goal is an authentic Egyptian experience, not a tourist bubble."""

# ---------------------------------------------------------------------------
# Embeddings + ChromaDB
# ---------------------------------------------------------------------------
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"

    def _embed(self, text):
        try:
            response = requests.post(self.url, json={
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]}
            })
            data = response.json()
            return data["embedding"]["values"]
        except Exception:
            return [0.0] * 768  # fallback: zero vector if embedding fails

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
# /ask endpoint — handles both "chat" and "plan" modes
# ---------------------------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    if request.is_json:
        data = request.get_json() or {}
        if not data:
            return jsonify({"type": "chat", "answer": "Invalid request.", "error": True}), 400
        request_type  = data.get("type", "chat")
        question      = data.get("question", "").strip()
        image_base64  = data.get("image", None)
        audio_base64  = data.get("audio", None)
        video_base64  = data.get("video", None)
        files         = data.get("files", [])
        lat           = data.get("lat", None)
        lng           = data.get("lng", None)
        history       = data.get("history", [])
    else:
        # Support multipart/form-data from Flutter
        request_type  = request.form.get("type", "chat")
        question      = request.form.get("question", "").strip()
        lat           = request.form.get("lat", None)
        lng           = request.form.get("lng", None)
        image_base64  = None
        audio_base64  = None
        video_base64  = None
        files         = []
        try:
            history = json.loads(request.form.get("history", "[]"))
        except Exception:
            history = []

    # Limit history to last 20 turns to avoid exceeding model context
    if len(history) > 20:
        history = history[-20:]

    # ===== PLAN MODE =======================================================
    if request_type == "plan":
        plan_prompt = question or "Create a 3-day Cairo itinerary"
        if lat and lng:
            plan_prompt += f" (user is at latitude {lat}, longitude {lng})"

        try:
            response = call_gemini(
                contents=plan_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=PLAN_SYSTEM,
                    response_mime_type="application/json",
                    temperature=0.6,
                    top_p=0.9,
                    max_output_tokens=4096,
                )
            )
            plan_data = json.loads(response.text)
        except json.JSONDecodeError:
            plan_data = {"title": "Trip Plan", "days": [], "error": "Failed to generate plan, please try again."}
        except Exception:
            return jsonify({"type": "plan", "data": {"title": "Trip Plan", "days": []}, "error": "All models are currently busy. Please try again in a moment."}), 503

        return jsonify({"type": "plan", "data": plan_data})

    # ===== CHAT MODE (default) =============================================

    # --- Location context ---------------------------------------------------
    location_context = ""
    if lat and lng:
        location_context = (
            f"The user is currently at GPS coordinates: "
            f"latitude {lat}, longitude {lng}. "
            f"Tailor all suggestions specifically to this exact location."
        )

    # --- Media parts --------------------------------------------------------
    media_parts = []
    media_labels = []

    if image_base64:
        try:
            media_parts.append(
                types.Part.from_bytes(data=base64.b64decode(image_base64), mime_type="image/jpeg")
            )
            media_labels.append("an image")
            if not question:
                question = "What is this? Identify it and give me useful information about it."
        except Exception:
            pass  # skip corrupted image

    if audio_base64:
        try:
            media_parts.append(
                types.Part.from_bytes(data=base64.b64decode(audio_base64), mime_type="audio/mp3")
            )
            media_labels.append("an audio clip")
            if not question:
                question = "Please listen to this audio and respond helpfully."
        except Exception:
            pass  # skip corrupted audio

    if video_base64:
        try:
            media_parts.append(
                types.Part.from_bytes(data=base64.b64decode(video_base64), mime_type="video/mp4")
            )
            media_labels.append("a video")
            if not question:
                question = "What can you tell me about this video?"
        except Exception:
            pass  # skip corrupted video

    # Generic files — any MIME type the model supports
    for f in files:
        try:
            mime = f.get("mime_type", "application/octet-stream")
            raw  = base64.b64decode(f.get("data", ""))
            media_parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
            media_labels.append(mime)
            if not question:
                ext = mime.split("/")[-1]
                question = f"Please analyze this {ext} file and provide relevant information."
        except Exception:
            continue  # skip corrupted file

    # 2. Handle actual uploaded files (multipart/form-data)
    for key, file in request.files.items():
        if file.filename == '':
            continue
        try:
            raw_bytes = file.read()
            mime_type = file.mimetype or "application/octet-stream"
            media_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
            media_labels.append(mime_type)
            if not question:
                if "image" in mime_type:
                    question = "What is this? Identify it and give me useful information about it."
                elif "audio" in mime_type:
                    question = "Please listen to this audio and respond helpfully."
                elif "video" in mime_type:
                    question = "What can you tell me about this video?"
                else:
                    ext = mime_type.split("/")[-1]
                    question = f"Please analyze this {ext} file and provide relevant information."
        except Exception as e:
            print(f"[WARN] Failed to process uploaded file: {e}")

    # --- RAG retrieval ------------------------------------------------------
    rag_context = ""
    if question:
        context_docs = retriever.invoke(question)
        rag_context = "\n\n".join([doc.page_content for doc in context_docs]).strip()

    # --- Build conversation history -----------------------------------------
    history_text = ""
    if history:
        lines = []
        for turn in history:
            role = "User" if turn.get("role") == "user" else "Soli"
            lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n".join(lines)

    # --- Live Search Context ------------------------------------------------
    live_search_context = ""
    if question and len(question) > 5 and request_type == "chat":
        # Only search if the question is substantial enough
        live_search_context = get_live_context(question)

    # --- Compose prompt -----------------------------------------------------
    sections = []

    if location_context:
        sections.append(f"[LOCATION]\n{location_context}")

    if rag_context:
        sections.append(f"[KNOWLEDGE BASE]\n{rag_context}")

    if live_search_context:
        sections.append(f"[LIVE WEB DATA]\n{live_search_context}\nUse this live data to answer if it's relevant.")

    if history_text:
        sections.append(f"[CONVERSATION SO FAR]\n{history_text}")

    if media_labels:
        sections.append(f"[MEDIA] The user has shared: {', '.join(media_labels)}")

    sections.append(f"User: {question}\nSoli:")

    prompt = "\n\n".join(sections)

    # --- Call Gemini --------------------------------------------------------
    contents = media_parts + [prompt]

    try:
        response = call_gemini(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=CHAT_SYSTEM,
                temperature=0.75,
                top_p=0.95,
                max_output_tokens=2048,
            )
        )
        answer = response.text.strip() if response.text else "I couldn't generate a response. Please try again."
        return jsonify({"type": "chat", "answer": answer})
    except Exception:
        return jsonify({"type": "chat", "answer": "I'm a bit overwhelmed right now, please try again in a moment.", "error": True}), 503

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 5000)))