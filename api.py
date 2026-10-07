import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ingest import ingest_uploaded_pdf
from retriever import retrieve
from reranker import rerank
from llm import generate_answer


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PDF AI Assistant API",
    description="RAG API for uploading PDFs and asking questions",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "PDF AI Assistant API is running",
        "docs": "/docs",
        "endpoints": [
            "/upload",
            "/upload-multiple",
            "/ask",
            "/health"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "RAG API is running"
    }


# ============================================================
# UPLOAD ONE PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # Check file type

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )


    temp_path = None

    try:

        # Create temporary file

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            content = await file.read()

            temp_file.write(content)

            temp_path = temp_file.name


        # Ingest PDF

        total_chunks = ingest_uploaded_pdf(
            temp_path,
            file.filename
        )


        return {
            "success": True,
            "filename": file.filename,
            "chunks_added": total_chunks,
            "message": "PDF processed successfully"
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # Remove temporary file

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)


# ============================================================
# UPLOAD MULTIPLE PDFs
# ============================================================

@app.post("/upload-multiple")
async def upload_multiple_pdfs(
    files: list[UploadFile] = File(...)
):

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No PDF files provided."
        )


    results = []

    total_chunks = 0


    for file in files:

        # Check extension

        if not file.filename.lower().endswith(".pdf"):

            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Only PDF files are allowed."
            })

            continue


        temp_path = None


        try:

            # Temporary file

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                content = await file.read()

                temp_file.write(content)

                temp_path = temp_file.name


            # Process PDF

            chunks = ingest_uploaded_pdf(
                temp_path,
                file.filename
            )


            total_chunks += chunks


            results.append({

                "filename": file.filename,

                "success": True,

                "chunks_added": chunks

            })


        except Exception as e:

            results.append({

                "filename": file.filename,

                "success": False,

                "error": str(e)

            })


        finally:

            # Remove temporary file

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                os.remove(temp_path)


    return {

        "success": True,

        "files_processed": len(results),

        "total_chunks_added": total_chunks,

        "files": results

    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()


    # Check question

    if not question:

        raise HTTPException(

            status_code=400,

            detail="Question cannot be empty."

        )


    try:

        # ----------------------------------------------------
        # STEP 1: RETRIEVE
        # ----------------------------------------------------

        docs = retrieve(
            question,
            top_k=10
        )


        # ----------------------------------------------------
        # CHECK DOCUMENTS
        # ----------------------------------------------------

        if not docs:

            return {

                "success": True,

                "question": question,

                "answer": (
                    "I could not find "
                    "relevant information "
                    "in the uploaded documents."
                )

            }


        # ----------------------------------------------------
        # STEP 2: RERANK
        # ----------------------------------------------------

        top_docs = rerank(

            question,

            docs,

            top_k=5

        )


        # ----------------------------------------------------
        # STEP 3: CREATE CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            top_docs
        )


        # ----------------------------------------------------
        # STEP 4: LLM
        # ----------------------------------------------------

        answer = generate_answer(

            question,

            context

        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "success": True,

            "question": question,

            "answer": answer,

            "documents_retrieved": len(docs),

            "documents_used": len(top_docs)

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )