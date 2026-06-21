"""
main.py
FastAPI backend exposing two endpoints:
  POST /upload  — accepts raw JD text, builds FAISS index, returns session_id
  POST /ask     — accepts question + session_id, returns LLM answer

Note: This file is optional for the Streamlit Cloud deployment (streamlit_app.py
calls the RAG pipeline directly), but it's useful if you want a separate
REST API you can test with Postman/curl, or plug into other frontends later.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid

from app.embeddings import process_jd_text
from app.rag_pipeline import ask_question

app = FastAPI(
    title="JobSense API",
    description="RAG-based Job Description Q&A backend",
    version="1.0.0"
)

# ── In-memory session store ───────────────────────────────────────────────────
# { session_id: { "index": faiss_index, "chunks": [...] } }
# For production, replace with Redis or a persistent store
sessions: dict = {}


# ── Request/Response models ───────────────────────────────────────────────────
class UploadRequest(BaseModel):
    jd_text: str          # Raw job description text


class UploadResponse(BaseModel):
    session_id: str       # Use this in subsequent /ask calls
    chunk_count: int      # How many chunks were created


class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "JobSense API is running. Use /upload and /ask endpoints."}


@app.post("/upload", response_model=UploadResponse)
def upload_jd(request: UploadRequest):
    """
    Accepts a job description as plain text.
    Builds FAISS index and stores it in session memory.
    Returns a session_id to use in /ask calls.
    """
    if not request.jd_text or len(request.jd_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description text is too short. Please provide the full JD."
        )

    # Build index
    faiss_index, chunks = process_jd_text(request.jd_text)

    # Store in session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "index": faiss_index,
        "chunks": chunks
    }

    return UploadResponse(session_id=session_id, chunk_count=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Accepts a question and session_id.
    Runs the RAG pipeline and returns the LLM's answer.
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a job description first."
        )

    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Question is too short."
        )

    session = sessions[request.session_id]

    answer = ask_question(
        question=request.question,
        faiss_index=session["index"],
        chunks=session["chunks"]
    )

    return AskResponse(answer=answer, session_id=request.session_id)