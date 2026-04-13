# ◈ DOCMIND — PDF Q&A Bot

> Generative AI-powered Retrieval-Augmented Generation (RAG) system for intelligent PDF-based question answering.

---

## 🧠 Tech Stack (Full Details)

| Component         | Technology                              | Details                                      |
|-------------------|-----------------------------------------|----------------------------------------------|
| **LLM**           | `llama-3.3-70b-versatile`               | Via Groq API (free tier) · 128k context       |
| **Embeddings**    | `BAAI/bge-small-en-v1.5`                | Local HuggingFace model · 384 dimensions     |
| **Vector DB**     | ChromaDB (persistent, local)            | Cosine similarity search · per-doc collections|
| **RAG Framework** | LangChain                               | ConversationalRetrievalChain + memory         |
| **PDF Loader**    | PyPDFLoader (LangChain Community)       | Page-level extraction with metadata           |
| **Text Splitter** | RecursiveCharacterTextSplitter          | 1000 tokens · 200 overlap                    |
| **Backend**       | Flask 3.0                               | REST API · session management                 |
| **Frontend**      | Vanilla HTML/CSS/JS                     | Dark editorial UI · drag-and-drop             |

---

## 📁 Project Structure

```
pdf-qa-bot/
├── app.py                    ← Flask app + all API routes
├── requirements.txt          ← Python dependencies
├── .env.example              ← Environment variables template
├── README.md
├── rag/
│   ├── __init__.py
│   ├── document_processor.py ← PDF loading + text chunking
│   ├── vector_store.py       ← ChromaDB embedding + retrieval
│   └── qa_chain.py          ← LangChain RAG chain + memory
├── static/
│   ├── css/style.css         ← Dark editorial UI styles
│   └── js/main.js            ← Chat logic, file upload, UI
├── templates/
│   └── index.html            ← Single-page frontend
├── uploads/                  ← Uploaded PDFs (auto-created)
└── chroma_db/                ← Persistent vector store (auto-created)
```

---

## ⚙️ Setup Instructions (Step-by-Step)

### Step 1 — Prerequisites

Make sure you have **Python 3.10+** installed:
```bash
python --version   # should be 3.10, 3.11, or 3.12
```

### Step 2 — Get a Free Groq API Key

1. Go to **https://console.groq.com**
2. Sign up for a free account (no credit card needed)
3. Navigate to **API Keys** → click **Create API Key**
4. Copy your key (starts with `gsk_...`)

> Groq's free tier gives ~14,400 requests/day — more than enough for development.

### Step 3 — Clone / Extract the Project

```bash
# If you have the zip, extract it, then:
cd pdf-qa-bot
```

### Step 4 — Create a Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows (CMD):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 5 — Install Dependencies

```bash
# Install CPU-only PyTorch first (saves ~2 GB vs GPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install all other dependencies
pip install -r requirements.txt
```

> ⏱ First install takes ~3–5 minutes. The `sentence-transformers` model (~90 MB) downloads on first run.

### Step 6 — Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Open .env and fill in your Groq API key
# On macOS/Linux:
nano .env
# On Windows:
notepad .env
```

Your `.env` should look like:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
FLASK_SECRET_KEY=any_random_string_here
FLASK_DEBUG=false
PORT=5000
```

### Step 7 — Run the Application

```bash
python app.py
```

You should see:
```
🚀  PDF Q&A Bot running on http://localhost:5000
```

### Step 8 — Use the App

1. Open **http://localhost:5000** in your browser
2. **Drag & drop** or **browse** to upload a PDF (up to 50 MB)
3. Wait for processing — you'll see: chunking → embedding → indexing
4. Once ready, **type any question** about your document
5. Ask follow-up questions — the AI remembers the conversation!

---

## 🔄 How RAG Works (Under the Hood)

```
PDF Upload
    │
    ▼
PyPDFLoader          → Extracts text page-by-page
    │
    ▼
RecursiveCharacterTextSplitter  → 1000 char chunks, 200 overlap
    │
    ▼
HuggingFace Embeddings          → Each chunk → 384-dim vector
    │
    ▼
ChromaDB                        → Vectors stored persistently

─────────────── At Query Time ───────────────

User Question
    │
    ▼
(If follow-up) Condense with chat history → self-contained query
    │
    ▼
ChromaDB similarity_search      → Top 4 most relevant chunks
    │
    ▼
LangChain prompt               → System prompt + context + question
    │
    ▼
LLaMA-3.3-70B (Groq)          → Reasoned, cited answer
    │
    ▼
ConversationBufferWindowMemory → Last 5 turns remembered
```

---

## 🔧 Configuration Options

Edit in the relevant files:

| Setting              | File                    | Default              | Description                          |
|----------------------|-------------------------|----------------------|--------------------------------------|
| `chunk_size`         | `app.py`                | 1000                 | Characters per chunk                 |
| `chunk_overlap`      | `app.py`                | 200                  | Overlap between chunks               |
| `k` (retrieval)      | `app.py`                | 4                    | Number of chunks retrieved per query |
| `LLM_MODEL`          | `rag/qa_chain.py`       | `llama-3.3-70b-versatile` | Groq model to use             |
| `LLM_TEMPERATURE`    | `rag/qa_chain.py`       | 0.2                  | Lower = more factual                 |
| `window_size`        | `rag/qa_chain.py`       | 5                    | Conversation turns to remember       |
| `EMBEDDING_MODEL`    | `rag/vector_store.py`   | `bge-small-en-v1.5`  | HuggingFace embedding model          |

---

## 🔁 Alternative LLM Options

If you prefer a different provider, swap in `rag/qa_chain.py`:

```python
# Option A: Anthropic Claude (claude-sonnet-4-20250514)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

# Option B: OpenAI GPT-4o
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Option C: Ollama (local, 100% free)
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2")  # Run: ollama pull llama3.2
```

---

## 🛠 Troubleshooting

| Issue | Fix |
|-------|-----|
| `GROQ_API_KEY not set` | Make sure `.env` exists and has your key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the venv |
| Slow first run | `sentence-transformers` model downloading (~90 MB) — normal |
| PDF extraction empty | Try a text-based PDF (not a scanned image PDF) |
| Port 5000 in use | Set `PORT=5001` in `.env` |
| ChromaDB error | Delete the `chroma_db/` folder and restart |

---

## 📊 API Endpoints

| Method | Endpoint       | Description                         |
|--------|----------------|-------------------------------------|
| `GET`  | `/`            | Frontend UI                         |
| `POST` | `/api/upload`  | Upload + process a PDF              |
| `POST` | `/api/chat`    | Send a question, get an answer      |
| `GET`  | `/api/history` | Fetch conversation history          |
| `POST` | `/api/reset`   | Clear session + delete document     |
| `GET`  | `/api/status`  | Health check + session status       |
