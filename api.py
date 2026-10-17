
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from ingest import ingest_uploaded_pdf
from retriever import retrieve
from reranker import rerank
from llm import generate_answer

app = FastAPI(
    title="PDF AI Assistant API",
    description="RAG API for uploading PDFs and asking questions",
    version="1.1.0",
)


class QuestionRequest(BaseModel):
    question: str
    session_id: str


@app.get("/")
def home():
    return {
        "message": "PDF AI Assistant API is running",
        "docs": "/docs",
        "endpoints": ["/upload", "/upload-multiple", "/ask", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    # Validate the file and session ID.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")

    temp_path = None

    try:
        # Save the uploaded file temporarily.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Store chunks with the session ID.
        total_chunks = ingest_uploaded_pdf(
            temp_path,
            file.filename,
            session_id,
        )

        return {
            "success": True,
            "filename": file.filename,
            "chunks_added": total_chunks,
            "message": "PDF processed successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/upload-multiple")
async def upload_multiple_pdfs(
    files: list[UploadFile] = File(...),
    session_id: str = Form(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files provided.")

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")

    results = []
    total_chunks = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Only PDF files are allowed.",
            })
            continue

        temp_path = None

        try:
            # Save this PDF temporarily.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name

            # Index the PDF under the current session.
            chunks = ingest_uploaded_pdf(
                temp_path,
                file.filename,
                session_id,
            )

            total_chunks += chunks

            results.append({
                "filename": file.filename,
                "success": True,
                "chunks_added": chunks,
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return {
        "success": all(item["success"] for item in results),
        "files_processed": len(results),
        "total_chunks_added": total_chunks,
        "files": results,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    session_id = request.session_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    try:
        # Retrieve only chunks belonging to this session.
        docs = retrieve(
            question,
            top_k=10,
            session_id=session_id,
        )

        # Do not ask the LLM to answer from unrelated documents.
        if not docs:
            return {
                "success": True,
                "question": question,
                "answer": (
                    "I could not find relevant information in the PDFs "
                    "uploaded in this browser session."
                ),
                "documents_retrieved": 0,
                "documents_used": 0,
            }

        # Rerank the retrieved chunks.
        top_docs = rerank(question, docs, top_k=5)

        # Build context for the LLM.
        context = "\n\n".join(top_docs)

        # Generate the answer from retrieved context.
        answer = generate_answer(question, context)

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "documents_retrieved": len(docs),
            "documents_used": len(top_docs),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
