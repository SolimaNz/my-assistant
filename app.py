try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("[INFO] Successfully swapped sqlite3 with pysqlite3-binary.")
except ImportError:
    print("[INFO] Using system sqlite3 (pysqlite3-binary not installed/needed).")

import os
import json
import base64
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from flask import Flask, request, jsonify
from flask_sock import Sock
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

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
# Utility — strip markdown code fences from Gemini JSON responses
# ---------------------------------------------------------------------------
def clean_json_text(text):
    """Strip markdown code fences from Gemini JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ---------------------------------------------------------------------------
# Live Search — DuckDuckGo (Free, Unlimited)
# ---------------------------------------------------------------------------
def get_live_context(query, lat=None, lng=None):
    """Searches DuckDuckGo and returns top 3 results as context.
    If GPS coordinates are available, appends them to localize the search."""
    try:
        search_query = query
        if lat and lng:
            search_query = f"{query} near latitude {lat}, longitude {lng}"
        results = DDGS().text(search_query, max_results=3)
        if results:
            context = "Live search results:\n"
            for r in results:
                context += f"- {r.get('title')}: {r.get('body')}\n"
            return context
    except Exception as e:
        print(f"[WARN] Live search failed: {e}")
    return ""

def needs_live_search(query):
    """Search ALWAYS, unless it's clearly a greeting or filler.
    This ensures every real question — in any language — gets live data."""
    query_lower = query.lower().strip("?.! ")

    # Small exclusion list — only skip search for obvious greetings/filler
    skip_phrases = {
        "hi", "hello", "hey", "hola", "salam", "ahlan", "howdy", "yo", "sup",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "who are you", "what are you", "what is your name",
        "who is soli", "tell me about yourself",
        "thanks", "thank you", "shukran", "merci",
        "bye", "goodbye", "see you", "ok", "okay",
        "مرحبا", "اهلا", "شكرا", "مع السلامة",
    }

    # Short filler (under 5 chars like "hi", "ok", "yo") or exact greeting match
    if query_lower in skip_phrases or len(query_lower) < 5:
        return False

    # Everything else → search for the best possible answer
    return True

CAIRO_TZ = ZoneInfo("Africa/Cairo")

def get_cairo_datetime():
    """Return current Cairo date, time, and timezone as a context string."""
    now = datetime.now(CAIRO_TZ)
    offset = now.strftime("%z")          # e.g. "+0300"
    offset_fmt = f"UTC{offset[:3]}:{offset[3:]}"  # e.g. "UTC+03:00"
    return (
        f"Current local time in Egypt: {now.strftime('%I:%M %p')} "
        f"on {now.strftime('%A, %B %d, %Y')} ({offset_fmt})"
    )

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
        audio_mime_hint = data.get("audio_mime", None)
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
        audio_mime_hint = None
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
        if history:
            prev = "\n".join(
                f"{'User' if t.get('role')=='user' else 'Soli'}: {t.get('content','')}"
                for t in history
            )
            plan_prompt = f"Previous conversation:\n{prev}\n\nNew request: {plan_prompt}"

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
            plan_data = json.loads(clean_json_text(response.text))
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
            img_bytes = base64.b64decode(image_base64)
            # Auto-detect image format from magic bytes
            if img_bytes[:4] == b'\x89PNG':
                img_mime = "image/png"
            elif len(img_bytes) > 11 and img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
                img_mime = "image/webp"
            elif img_bytes[:3] == b'GIF':
                img_mime = "image/gif"
            else:
                img_mime = "image/jpeg"
            media_parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type=img_mime)
            )
            media_labels.append("an image")
            if not question:
                question = "What is this? Identify it and give me useful information about it."
        except Exception:
            pass  # skip corrupted image

    if audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            # Use explicit hint from Flutter, or auto-detect from magic bytes
            if audio_mime_hint:
                audio_mime = audio_mime_hint
            elif audio_bytes[:4] == b'fLaC':
                audio_mime = "audio/flac"
            elif audio_bytes[:4] == b'RIFF':
                audio_mime = "audio/wav"
            elif audio_bytes[:4] == b'OggS':
                audio_mime = "audio/ogg"
            elif len(audio_bytes) > 7 and audio_bytes[4:8] == b'ftyp':
                audio_mime = "audio/mp4"  # covers .m4a / .aac containers
            elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                audio_mime = "audio/mp3"
            else:
                audio_mime = "audio/mp3"  # fallback
            media_parts.append(
                types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime)
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
            
            # Auto-detect image or audio MIME type if generic
            if mime_type in ("application/octet-stream", "application/x-binary"):
                if raw_bytes[:4] == b'\x89PNG':
                    mime_type = "image/png"
                elif len(raw_bytes) > 11 and raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
                    mime_type = "image/webp"
                elif raw_bytes[:3] == b'GIF':
                    mime_type = "image/gif"
                elif raw_bytes[:2] == b'\xff\xd8':
                    mime_type = "image/jpeg"
                elif raw_bytes[:4] == b'fLaC':
                    mime_type = "audio/flac"
                elif raw_bytes[:4] == b'RIFF':
                    mime_type = "audio/wav"
                elif raw_bytes[:4] == b'OggS':
                    mime_type = "audio/ogg"
                elif len(raw_bytes) > 7 and raw_bytes[4:8] == b'ftyp':
                    mime_type = "audio/mp4"
                elif raw_bytes[:3] == b'ID3' or raw_bytes[:2] == b'\xff\xfb':
                    mime_type = "audio/mp3"

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
    time_context = ""
    if question and needs_live_search(question):
        live_search_context = get_live_context(question, lat, lng)
        time_context = get_cairo_datetime()

    # --- Compose prompt -----------------------------------------------------
    sections = []

    if location_context:
        sections.append(f"[LOCATION]\n{location_context}")

    if rag_context:
        sections.append(f"[KNOWLEDGE BASE]\n{rag_context}")

    if time_context:
        sections.append(f"[CURRENT TIME & DATE]\n{time_context}")

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

# ---------------------------------------------------------------------------
# /health endpoint — monitoring
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ---------------------------------------------------------------------------
# /voice WebSocket — real-time voice call proxy (Flutter <-> Gemini Live API)
# ---------------------------------------------------------------------------
SOLI_VOICE_SYSTEM = """You are Soli, a smart and friendly virtual assistant for anyone in Egypt.
You're not just a tour guide — you're a local expert who helps with EVERYTHING:
tourism, daily life, emergencies, language, transportation, shopping, food,
health, legal questions, scams, cultural tips, and anything else someone
in Egypt might need help with.

Core rules:
- Keep your spoken responses concise and natural — this is a live voice call,
  not a text chat. Nobody wants to listen to a 5-paragraph essay.
- Match the user's energy. Short question → short answer. Deep question → deep answer.
- Talk like a real person having a conversation. Be warm, helpful, and a little witty.
- Always reply in the same language the user speaks.
- Share insider tips, local slang, price expectations, and safety heads-ups
  when they're relevant — don't force them into every answer.
- If someone is in trouble (lost passport, medical issue, police),
  give clear and calm step-by-step guidance.
- Go beyond typical tourist spots. Recommend the real local Egyptian experience.
- If you don't know something, say so honestly rather than making it up.
- Never start your response with "Great question!" or similar filler.
- Remember: you are speaking out loud, so use natural speech patterns,
  pauses, and conversational flow."""

@sock.route('/voice')
def voice_stream(ws):
    """WebSocket proxy: Flutter <-> Gemini Live API for real-time voice."""
    print("[Voice] Client connected")

    async def _run():
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SOLI_VOICE_SYSTEM,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"
                    )
                )
            ),
        )

        model = "gemini-2.5-flash-native-audio-latest"

        async with client.aio.live.connect(model=model, config=live_config) as session:
            # Notify Flutter that the voice session is ready
            ws.send(b"READY")
            print("[Voice] Gemini session ready, streaming started")

            stop_event = asyncio.Event()

            async def forward_to_gemini():
                """Read audio bytes from Flutter WebSocket, forward to Gemini."""
                while not stop_event.is_set():
                    try:
                        # Run blocking ws.receive in a thread so it doesn't freeze the event loop
                        data = await asyncio.to_thread(ws.receive, 0.05)
                    except Exception:
                        # Connection closed or error — stop
                        stop_event.set()
                        break

                    if data is None:
                        # Timeout — no data yet, just keep waiting
                        continue

                    if isinstance(data, bytes) and len(data) > 0:
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=data,
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                        except Exception:
                            stop_event.set()
                            break

            async def forward_to_flutter():
                """Read audio response from Gemini, send back to Flutter."""
                while not stop_event.is_set():
                    try:
                        async for response in session.receive():
                            if stop_event.is_set():
                                break
                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        ws.send(part.inline_data.data)
                    except Exception:
                        stop_event.set()
                        break

            await asyncio.gather(
                forward_to_gemini(),
                forward_to_flutter()
            )

    try:
        asyncio.run(_run())
    except Exception as e:
        print(f"[Voice] Session ended: {e}")
    finally:
        print("[Voice] Client disconnected")

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 5000)))