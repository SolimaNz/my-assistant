# 🇪🇬 Soli: Egypt's Smart Virtual Assistant (Backend)

Soli is a production-ready, highly responsive Flask backend that powers an intelligent, context-aware virtual assistant designed for residents, expats, and tourists in Egypt. 

Unlike generic chatbots, Soli is engineered to act as a **digital local companion**—combining multimodal inputs, dynamic live web search, timezone-aware clocks, custom localized knowledge bases (RAG), and a low-latency voice streaming interface to provide an authentic, warm, and highly practical Egyptian experience.

---

##  Extended Project Overview

Traveling or living in Egypt often presents unique challenges—from understanding local pricing and navigation to cultural customs, local slang, transportation logistics, and emergency services. 

Soli solves this by acting as a single, comprehensive gateway. Whether a user wants to identify a historical monument from a photo, plan a complete budget-friendly itinerary, ask about local tipping culture (المنظرة والتبشيش), find nearby pharmacies, or speak directly to a local expert in real-time Arabic or English, Soli responds with accurate, localized, and context-aware intelligence.

---

##  Core Capabilities

### 1. Multimodal Context Ingestion (`/ask`)
Soli doesn't just read text; it understands the world around it. The backend accepts and parses:
- **Images:** Identifies historical landmarks, menus, signs, or products, and provides useful local context.
- **Audio Clips:** Listens to user questions or environment audio and translates or responds directly.
- **Videos & Files:** Processes and analyzes video loops or text documents for deep-dive analysis.
- **Automatic MIME Recovery:** Uses file signature magic-byte checks to rebuild binary objects even if client-side MIME headers are missing or generic.

### 2. Location-Aware Personalization
By passing GPS coordinates (`lat`/`lng`) from the frontend client, Soli automatically:
- Calculates distances and customizes recommendations (e.g. nearby local food carts, pharmacies, ATMs, and transport options) specific to that exact spot.
- **Localizes live web searches** by appending the user's coordinates to the search query, ensuring results are relevant to the user's actual location in Egypt — not the server's data center.

### 3. Always-On Live Web Search
Soli performs a live DuckDuckGo web search on **every real question** — in any language — to ensure the most accurate, up-to-date answers. Only trivial greetings and filler (e.g. "hi", "thanks", "bye") are excluded to avoid polluting context with irrelevant search results.

This approach:
* **Works in every language.** French, German, Chinese, Arabic, Spanish — all get live data without maintaining language-specific keyword lists.
* **Never misses a real query.** No keyword gaps or edge cases to chase.
* **Adds minimal latency.** DuckDuckGo searches average ~1.3 seconds, which is negligible for a travel assistant where accuracy matters more than milliseconds.

### 4. Smart Clock Context Injection (DST-Aware)
LLMs have no concept of real-time clocks. Soli resolves this by injecting a precise, timezone-aware timestamp adjusted dynamically to **Egypt's timezone (`Africa/Cairo`)**. The clock automatically accounts for **Daylight Saving Time (DST)** changes (UTC+3 in summer, UTC+2 in winter) so that all queries about "today", "now", or specific hourly schedules remain accurate in production.

### 5. Structured Trip Planner (JSON Engine)
When requested, Soli switches context to a specialized planning agent. Using structured schemas, it generates a complete day-by-day JSON response containing:
- **Sequential activity IDs** (`id: 1, 2, 3…`) injected per day for direct use in frontend card displays.
- Activities sorted logically by geographic coordinates to minimize travel time.
- Accurate coordinates (`lat` and `lng`) for mapping on the frontend.
- Parallel pricing estimates in both **USD** and **EGP**.
- Recommendations for both major landmarks and local, off-the-beaten-path Egyptian experiences (محلات الكشري، القهاوي البلدي، والأسواق الشعبية).

### 6. Low-Latency WebSocket Voice Proxy (`/voice`)
Soli provides a bi-directional WebSocket interface mapping to Gemini's native audio engine (`gemini-2.5-flash-native-audio-latest`). 
- Handles real-time voice streaming with sub-second response latency.
- Features a custom asynchronous event loop running concurrently within Flask's worker threads.
- Streams audio input/output seamlessly using raw 16kHz PCM audio bytes.

---

##  System Architecture

```
                       +-------------------+
                       |  Frontend Client  |
                       | (Flutter / Web)   |
                       +---------+---------+
                                 |
        +------------------------+------------------------+
        | HTTP Requests (/ask)                            | WebSocket Stream (/voice)
        v                                                 v
+-------+------------------+                    +---------+------------------+
| Flask HTTP App           |                    | Flask-Sock WebSocket App   |
+-------+------------------+                    +---------+------------------+
        |                                                 |
        |-- 1. Parse GPS, Media & History                 |-- 1. Initialize Gemini Live Session
        |-- 2. Fetch RAG Context (ChromaDB)               |-- 2. Map Asynchronous Event Loop
        |-- 3. Always-On Live Web Search (GPS-aware)      |-- 3. Stream Bi-directional 16kHz PCM
        |-- 4. Inject Timezone Clock (Cairo EEST)         |
        v                                                 v
+-------+-------------------------------------------------+------------------+
|                              Google GenAI SDK                              |
|   (gemini-3.1-flash-lite / gemini-2.5-flash / gemini-2.5-flash-audio)      |
+----------------------------------------------------------------------------+
```

---

##  Setup & Local Development

### 1. Installation
Clone the repository and install the required dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Configure the following keys in your local directory:
```env
GEMINI_API_KEY=your_gemini_api_key
CHROMA_DIR=chroma_db
PORT=5000
```

### 3. RAG Initialization
Drop your knowledge files into the `data/` folder and run the setup script to embed them into ChromaDB:
```bash
python rag_setup.py
```
Supported file types: `.txt`, `.md`, `.pdf`, `.csv`, `.json`, `.docx`, `.html`

The script auto-discovers all files in `data/` and uses **incremental embedding** — only new or modified files are processed. Unchanged files are skipped automatically, saving API quota. Deleted files have their embeddings removed from ChromaDB.

### 4. Running Locally
```bash
python app.py
```

### 5. Running Tests
Run the full test suite before pushing:
```bash
python -m unittest test_all -v
```
---

##  Azure App Service Deployment

Soli is designed for deployment on **Azure App Service (Linux)**.

### Configuration Checklist:
1. **WebSockets:** Under **Configuration > General Settings**, ensure **WebSockets is ON**.
2. **Environment Variables:** Define `GEMINI_API_KEY` and `SCM_DO_BUILD_DURING_DEPLOYMENT = true` in the portal.
3. **Startup Command:** Set the startup command under **Stack settings** to:
   ```bash
   export ANTENV=$(find /tmp -name "antenv" -type d | head -n 1); cp $ANTENV/lib/python3.11/site-packages/pysqlite3/_sqlite3*.so $ANTENV/lib/python3.11/site-packages/pysqlite3/pysqlite3.so || true; export LD_PRELOAD=$(find /tmp -name pysqlite3.so | head -n 1); gunicorn --bind=0.0.0.0 --timeout 600 --workers 1 --threads 50 app:app
   ```
