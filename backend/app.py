"""
DocuChat API — a document-grounded support chatbot (RAG) with streaming answers.

Design goals (mirrors a real production support bot):
  * Answers ONLY from the uploaded documents, and cites where each answer came from.
  * Never invents facts: if nothing relevant is found, it says so.
  * Streams the answer token-by-token (SSE) for a "typing like ChatGPT" feel.
  * Collects 👍 / 👎 feedback and exposes simple admin stats.

Retrieval is a dependency-light TF-IDF cosine search, so the demo runs for free
with no API keys. If OPENAI_API_KEY is set, the same retrieved context is handed
to an LLM for a nicer, generated answer — otherwise we return an extractive answer
built from the best-matching sentences. Either path streams and cites sources.
"""

import asyncio
import json
import os
import re
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Swagger UI moved to /api-docs so our own GET /docs (list documents) isn't
# shadowed by FastAPI's built-in interactive docs route.
app = FastAPI(title="DocuChat API", version="1.0.0", docs_url="/api-docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo; lock to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# In-memory store (a demo; swap for Postgres/pgvector in production)
# --------------------------------------------------------------------------- #
DOCS: List[dict] = []            # {id, title, text}
CHUNKS: List[dict] = []          # {doc_id, title, text}
QUESTIONS: List[dict] = []       # {id, q, answer, sources, grounded, ts}
FEEDBACK = {"up": 0, "down": 0}

_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None
MIN_SCORE = 0.06                 # below this we treat retrieval as "no answer"


def _chunk(text: str, size: int = 90, overlap: int = 20) -> List[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += size - overlap
    return out


def _reindex() -> None:
    global _vectorizer, _matrix, CHUNKS
    CHUNKS = [
        {"doc_id": d["id"], "title": d["title"], "text": c}
        for d in DOCS
        for c in _chunk(d["text"])
    ]
    if CHUNKS:
        _vectorizer = TfidfVectorizer(stop_words="english")
        _matrix = _vectorizer.fit_transform([c["text"] for c in CHUNKS])
    else:
        _vectorizer, _matrix = None, None


def _retrieve(query: str, k: int = 3):
    if not CHUNKS or _vectorizer is None:
        return []
    q = _vectorizer.transform([query])
    sims = cosine_similarity(q, _matrix)[0]
    order = np.argsort(sims)[::-1][:k]
    return [(CHUNKS[i], float(sims[i])) for i in order if sims[i] >= MIN_SCORE]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _extractive_answer(query: str, hits) -> str:
    """Build an answer from the most query-relevant sentences of the top hits."""
    q_terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    scored = []
    for chunk, _ in hits:
        for s in _sentences(chunk["text"]):
            overlap = len(q_terms & {w for w in re.findall(r"[a-z0-9]+", s.lower())})
            if overlap:
                scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [s for _, s in scored[:2]] or [hits[0][0]["text"][:280]]
    return " ".join(picked)


async def _llm_stream(query: str, context: str):
    """Optional real-LLM path (OpenAI). Yields text deltas. Falls back on error."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return
    try:
        from openai import OpenAI  # imported lazily; only needed if a key is set

        client = OpenAI(api_key=key)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't have that "
            "information. Be concise.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        stream = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2,
        )
        for part in stream:
            delta = part.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception:
        return  # caller falls back to extractive


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #
class ChatIn(BaseModel):
    message: str


class DocIn(BaseModel):
    title: str
    text: str


class FeedbackIn(BaseModel):
    vote: str  # "up" | "down"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"ok": True, "docs": len(DOCS), "chunks": len(CHUNKS)}


@app.get("/docs")
def list_docs():
    return [{"id": d["id"], "title": d["title"], "chars": len(d["text"])} for d in DOCS]


@app.post("/docs")
def add_doc(doc: DocIn):
    if not doc.text.strip():
        raise HTTPException(400, "Empty document")
    d = {"id": str(uuid.uuid4())[:8], "title": doc.title.strip() or "Untitled", "text": doc.text.strip()}
    DOCS.append(d)
    _reindex()
    return {"id": d["id"], "title": d["title"]}


@app.delete("/docs/{doc_id}")
def del_doc(doc_id: str):
    global DOCS
    DOCS = [d for d in DOCS if d["id"] != doc_id]
    _reindex()
    return {"ok": True}


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXT = {".pdf", ".txt", ".md"}


@app.post("/docs/upload")
async def upload_doc(file: UploadFile):
    """Add a document from an uploaded file (.pdf, .txt or .md)."""
    name = file.filename or "upload"
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .pdf, .txt or .md")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File too large (max 5 MB)")

    if ext == ".pdf":
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            raise HTTPException(400, "Could not read this PDF")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 40:
        raise HTTPException(
            400,
            "No readable text found in the file (scanned/image-only PDFs need OCR, which the demo doesn't include)",
        )

    d = {"id": str(uuid.uuid4())[:8], "title": os.path.splitext(name)[0][:80], "text": text}
    DOCS.append(d)
    _reindex()
    return {"id": d["id"], "title": d["title"], "chars": len(text)}


@app.post("/chat")
async def chat(inp: ChatIn):
    query = inp.message.strip()
    if not query:
        raise HTTPException(400, "Empty message")
    hits = _retrieve(query)
    sources = [{"title": h["title"], "score": round(s, 3)} for h, s in hits]

    async def gen():
        record = {"id": str(uuid.uuid4())[:8], "q": query, "ts": time.time()}
        if not hits:
            msg = "I couldn't find anything about that in the documents I have. Could you rephrase, or ask about our shipping, returns, or account topics?"
            for w in msg.split(" "):
                yield f"data: {json.dumps({'delta': w + ' '})}\n\n"
                await asyncio.sleep(0.02)
            record.update(answer=msg, sources=[], grounded=False)
            QUESTIONS.append(record)
            yield f"data: {json.dumps({'done': True, 'sources': [], 'grounded': False})}\n\n"
            return

        context = "\n\n".join(f"[{h['title']}] {h['text']}" for h, _ in hits)
        collected = ""

        used_llm = False
        async for delta in _llm_stream(query, context):
            used_llm = True
            collected += delta
            yield f"data: {json.dumps({'delta': delta})}\n\n"

        if not used_llm:  # free/extractive fallback
            answer = _extractive_answer(query, hits)
            for w in answer.split(" "):
                collected += w + " "
                yield f"data: {json.dumps({'delta': w + ' '})}\n\n"
                await asyncio.sleep(0.025)

        record.update(answer=collected.strip(), sources=sources, grounded=True)
        QUESTIONS.append(record)
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'grounded': True})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# Telegram interface — same RAG brain and knowledge base, chat via Telegram.
# Set TELEGRAM_BOT_TOKEN (+ optional TELEGRAM_WEBHOOK_SECRET) and register the
# webhook: https://api.telegram.org/bot<token>/setWebhook?url=<api>/telegram/webhook
# --------------------------------------------------------------------------- #
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

TG_START_TEXT = (
    "Hi! I'm DocuChat — a document-grounded support bot. I answer ONLY from my "
    "knowledge base (a demo store: shipping, returns, accounts) and I cite my sources.\n\n"
    "Try asking:\n"
    "• How long does shipping take?\n"
    "• What's your return policy?\n"
    "• How do I reset my password?\n\n"
    "If it's not in my documents, I'll say so instead of making something up.\n"
    "Web version + admin panel: https://yagami-reverse.github.io/docuchat/"
)


async def _tg_send(chat_id: int, text: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
        )


@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    if TG_SECRET and req.headers.get("x-telegram-bot-api-secret-token") != TG_SECRET:
        raise HTTPException(403, "Bad webhook secret")
    if not TG_TOKEN:
        return {"ok": True}

    update = await req.json()
    msg = update.get("message") or update.get("edited_message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id or not text:
        return {"ok": True}

    if text.startswith("/start") or text.startswith("/help"):
        await _tg_send(chat_id, TG_START_TEXT)
        return {"ok": True}

    hits = _retrieve(text)
    record = {"id": str(uuid.uuid4())[:8], "q": f"[tg] {text}", "ts": time.time()}
    if not hits:
        answer = (
            "I couldn't find anything about that in my documents. "
            "Try asking about shipping, returns, or account topics."
        )
        record.update(answer=answer, sources=[], grounded=False)
        QUESTIONS.append(record)
        await _tg_send(chat_id, answer)
        return {"ok": True}

    context = "\n\n".join(f"[{h['title']}] {h['text']}" for h, _ in hits)
    answer = ""
    async for delta in _llm_stream(text, context):
        answer += delta
    if not answer:
        answer = _extractive_answer(text, hits)

    sources = [{"title": h["title"], "score": round(s, 3)} for h, s in hits]
    record.update(answer=answer, sources=sources, grounded=True)
    QUESTIONS.append(record)

    await _tg_send(chat_id, f"{answer}\n\n\U0001F4C4 Sources: " + ", ".join(h["title"] for h, _ in hits))
    return {"ok": True}


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    if fb.vote not in ("up", "down"):
        raise HTTPException(400, "vote must be 'up' or 'down'")
    FEEDBACK[fb.vote] += 1
    return {"ok": True, **FEEDBACK}


@app.get("/questions")
def questions(limit: int = 50):
    return list(reversed(QUESTIONS[-limit:]))


@app.get("/stats")
def stats():
    total = FEEDBACK["up"] + FEEDBACK["down"]
    return {
        "documents": len(DOCS),
        "questions_asked": len(QUESTIONS),
        "grounded": sum(1 for q in QUESTIONS if q.get("grounded")),
        "unanswered": sum(1 for q in QUESTIONS if not q.get("grounded")),
        "feedback": FEEDBACK,
        "satisfaction": round(100 * FEEDBACK["up"] / total) if total else None,
    }


# --------------------------------------------------------------------------- #
# Seed data — a fictional store, matching the "FAQ / shipping / returns" domain
# --------------------------------------------------------------------------- #
def _seed():
    DOCS.extend(
        [
            {
                "id": "shipping",
                "title": "Shipping Policy",
                "text": (
                    "We ship worldwide. Orders are processed within 1-2 business days. "
                    "Standard shipping takes 3-5 business days within the US and 7-14 "
                    "business days internationally. Express shipping (1-2 business days) "
                    "is available at checkout for an extra fee. Shipping is free on all "
                    "orders over 50 dollars. Once your order ships you receive a tracking "
                    "link by email. We currently cannot ship to PO boxes."
                ),
            },
            {
                "id": "returns",
                "title": "Return & Refund Policy",
                "text": (
                    "You can return most items within 30 days of delivery for a full "
                    "refund. Items must be unused and in the original packaging. To start "
                    "a return, open your account, go to Orders, and select Return. Refunds "
                    "are issued to the original payment method within 5-7 business days "
                    "after we receive the item. Final-sale and personalized items cannot be "
                    "returned. Return shipping is free for defective items; otherwise a "
                    "5 dollar return label fee applies."
                ),
            },
            {
                "id": "account",
                "title": "Account & Orders FAQ",
                "text": (
                    "To reset your password, click 'Forgot password' on the sign-in page "
                    "and follow the email link. You can change your shipping address before "
                    "an order ships from Orders > Edit. To cancel an order, contact support "
                    "within 1 hour of placing it. We accept Visa, Mastercard, and PayPal. "
                    "Gift cards never expire and can be combined with one promo code per order."
                ),
            },
        ]
    )
    _reindex()


_seed()
