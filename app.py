"""
PDF Q&A Bot — Flask Application (Large-PDF Edition)
====================================================
Key upgrades vs v1
──────────────────
• Upload is non-blocking: returns a job_id immediately.
• Processing runs in a background thread.
• Real-time progress is streamed to the browser via Server-Sent Events (SSE).
• PDF pages are streamed in batches (never fully loaded into RAM).
• Chunks are embedded in sub-batches of 500 (safe for any PDF size).
• Upload size limit raised to 500 MB.
"""

import os
import uuid
import json
import time
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStoreManager
from rag.qa_chain import build_qa_chain, create_memory

# ── App Config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32))

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024   # 500 MB
os.environ["ANONYMIZED_TELEMETRY"] = "False"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ── In-memory stores ──────────────────────────────────────────────────────────
# Chat sessions: session_id → {chain, memory, filename, stats, messages, …}
chat_sessions: dict = {}

# Processing jobs: job_id → {status, pct, message, error, session_id}
jobs: dict = {}

# Lock for jobs dict (written from background threads)
jobs_lock = threading.Lock()

# ── Singletons ────────────────────────────────────────────────────────────────
processor  = DocumentProcessor(chunk_size=1000, chunk_overlap=200, page_batch_size=30)
vs_manager = VectorStoreManager()


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_or_create_session() -> str:
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def set_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id].update(kwargs)

def format_sources(source_docs: list) -> list:
    seen, sources = set(), []
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        file = doc.metadata.get("source_file", "document")
        key  = (file, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    file,
                "page":    int(page) + 1 if isinstance(page, int) else page,
                "snippet": doc.page_content[:200].strip() + "…",
            })
    return sources


# ── Background processing ─────────────────────────────────────────────────────
def process_pdf(job_id: str, session_id: str, save_path: Path, filename: str):
    """
    Runs in a daemon thread. Streams PDF page-batches → embeds in chunks →
    writes into ChromaDB. Updates `jobs[job_id]` at every step so SSE can
    relay progress to the browser.
    """
    try:
        key = f"{session_id}_{filename}"

        # ── Step 1: count pages ──────────────────────────────────────────────
        set_job(job_id, status="running", pct=2, message="Counting pages…")
        total_pages = processor.count_pages(str(save_path))

        # ── Step 2: stream + embed ───────────────────────────────────────────
        vectordb       = None
        all_chunks     = []
        pages_done     = 0
        embed_progress = [0]        # mutable cell for inner callback

        def embed_cb(done, total):
            embed_progress[0] = done
            # pct: pages 5–85, embedding within each batch
            base = 5 + int(80 * pages_done / max(total_pages, 1))
            pct  = base + int(10 * done / max(total, 1))
            set_job(job_id, pct=pct,
                    message=f"Embedding chunks ({done}/{total} in batch)…")

        for batch_chunks, pages_done, _ in processor.stream_chunks(str(save_path)):
            all_chunks.extend(batch_chunks)

            # Embed this page-batch incrementally
            vectordb = vs_manager.ingest_batch(
                batch_chunks, key,
                vectordb=vectordb,
                progress_cb=embed_cb,
            )

            page_pct = 5 + int(80 * pages_done / max(total_pages, 1))
            set_job(job_id, pct=page_pct,
                    message=f"Processed {pages_done}/{total_pages} pages…")

        # ── Step 3: build QA chain ───────────────────────────────────────────
        set_job(job_id, pct=90, message="Building retrieval chain…")
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},     # slightly more chunks for large docs
        )
        memory = create_memory(session_id)
        chain  = build_qa_chain(retriever, memory)

        stats = processor.get_stats(all_chunks)
        stats["total_pages"] = total_pages

        chat_sessions[session_id] = {
            "chain":        chain,
            "memory":       memory,
            "filename":     filename,
            "save_path":    str(save_path),
            "stats":        stats,
            "uploaded_at":  datetime.utcnow().isoformat(),
            "messages":     [],
            "collection_key": key,
        }

        set_job(job_id, status="done", pct=100,
                message=f"Ready! {stats['total_chunks']} chunks · {total_pages} pages",
                stats=stats, session_id=session_id)

    except Exception as exc:
        set_job(job_id, status="error", pct=0, message="Processing failed.",
                error=str(exc))
        if save_path.exists():
            save_path.unlink(missing_ok=True)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """
    Accept a PDF, save it, spawn background processing, return job_id.
    The client polls /api/progress/<job_id> (SSE) for updates.
    """
    session_id = get_or_create_session()

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filename  = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
    file.save(save_path)

    job_id = str(uuid.uuid4())
    set_job(job_id, status="queued", pct=0, message="Queued…", session_id=session_id)

    # Launch background thread
    t = threading.Thread(
        target=process_pdf,
        args=(job_id, session_id, save_path, filename),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "filename": filename})


@app.route("/api/progress/<job_id>")
def progress_stream(job_id: str):
    """
    SSE endpoint — streams job progress events until done/error.
    The browser receives: data: {"pct":45,"message":"…","status":"running"}
    """
    def generate():
        sent_done = False
        while not sent_done:
            with jobs_lock:
                job = dict(jobs.get(job_id, {}))

            payload = json.dumps({
                "pct":     job.get("pct", 0),
                "message": job.get("message", "Starting…"),
                "status":  job.get("status", "queued"),
                "stats":   job.get("stats"),
                "error":   job.get("error"),
            })
            yield f"data: {payload}\n\n"

            if job.get("status") in ("done", "error"):
                sent_done = True
            else:
                time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # nginx: disable proxy buffering
        },
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    session_id = get_or_create_session()
    data       = request.get_json(silent=True) or {}
    question   = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    if session_id not in chat_sessions:
        return jsonify({"error": "No document loaded. Upload a PDF first."}), 400

    sess = chat_sessions[session_id]
    try:
        result       = sess["chain"].invoke({"question": question})
        answer       = result.get("answer", "")
        source_docs  = result.get("source_documents", [])
        sources      = format_sources(source_docs)

        sess["messages"].append({"role": "user", "content": question})
        sess["messages"].append({"role": "assistant", "content": answer, "sources": sources})

        return jsonify({"answer": answer, "sources": sources, "model": "llama-3.3-70b-versatile"})

    except Exception as exc:
        return jsonify({"error": f"LLM error: {exc}"}), 500


@app.route("/api/history")
def history():
    session_id = get_or_create_session()
    if session_id not in chat_sessions:
        return jsonify({"messages": [], "filename": None, "stats": None})
    sess = chat_sessions[session_id]
    return jsonify({
        "messages": sess["messages"],
        "filename": sess.get("filename"),
        "stats":    sess.get("stats"),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    session_id = get_or_create_session()
    if session_id in chat_sessions:
        sess = chat_sessions.pop(session_id)
        path = Path(sess.get("save_path", ""))
        if path.exists():
            path.unlink(missing_ok=True)
        vs_manager.delete_collection(sess.get("collection_key", ""))
    return jsonify({"success": True})


@app.route("/api/status")
def status():
    session_id = get_or_create_session()
    has_doc    = session_id in chat_sessions
    return jsonify({
        "status":       "ok",
        "has_document": has_doc,
        "filename":     chat_sessions[session_id]["filename"] if has_doc else None,
    })


if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n🚀  PDF Q&A Bot (large-PDF edition) → http://localhost:{port}\n")
    # threaded=True is required for SSE to work alongside regular requests
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
