import os
import tempfile

import streamlit as st

from ingest import ingest_uploaded_pdf
from retriever import retrieve
from reranker import rerank
from llm import generate_answer


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("📚 PDF AI Assistant")

st.write(
    "Upload multiple PDFs and ask questions from them."
)


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "processed_files" not in st.session_state:

    st.session_state.processed_files = []


# ==================================================
# MULTIPLE PDF UPLOAD
# ==================================================

st.subheader("📄 Upload PDFs")


uploaded_files = st.file_uploader(

    "Choose PDF files",

    type=["pdf"],

    accept_multiple_files=True
)


# ==================================================
# PROCESS PDFs
# ==================================================

if uploaded_files:

    st.write(
        f"📚 {len(uploaded_files)} PDF(s) selected"
    )


    # Show selected files

    for file in uploaded_files:

        st.write(
            f"📄 {file.name}"
        )


    # Process button

    if st.button("🚀 Process All PDFs"):

        total_chunks = 0

        progress_bar = st.progress(0)

        status_text = st.empty()


        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            temp_path = None

            try:

                status_text.write(
                    f"Processing "
                    f"{uploaded_file.name}..."
                )


                # --------------------------------------
                # CREATE TEMP FILE
                # --------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                # --------------------------------------
                # INGEST PDF
                # --------------------------------------

                chunks = ingest_uploaded_pdf(

                    temp_path,

                    uploaded_file.name

                )


                total_chunks += chunks


                # --------------------------------------
                # SAVE PROCESSED FILE
                # --------------------------------------

                if (
                    uploaded_file.name
                    not in st.session_state.processed_files
                ):

                    st.session_state.processed_files.append(

                        uploaded_file.name

                    )


                # --------------------------------------
                # DELETE TEMP FILE
                # --------------------------------------

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(
                        temp_path
                    )


                # --------------------------------------
                # UPDATE PROGRESS
                # --------------------------------------

                progress = (
                    (index + 1)
                    / len(uploaded_files)
                )

                progress_bar.progress(
                    progress
                )


            except Exception as e:

                st.error(
                    f"❌ Error processing "
                    f"{uploaded_file.name}: {e}"
                )


            finally:

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(
                        temp_path
                    )


        # ------------------------------------------
        # COMPLETE
        # ------------------------------------------

        status_text.write(
            "✅ All PDFs processed!"
        )


        st.success(

            f"✅ {len(uploaded_files)} PDF(s) "
            f"processed successfully.\n\n"

            f"📦 Total chunks added: "
            f"{total_chunks}"

        )


# ==================================================
# PROCESSED PDFs
# ==================================================

if st.session_state.processed_files:

    st.subheader(
        "📚 Processed PDFs"
    )


    for file in st.session_state.processed_files:

        st.write(
            f"✅ {file}"
        )


# ==================================================
# CHAT HISTORY
# ==================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )


# ==================================================
# QUESTION
# ==================================================

query = st.chat_input(
    "Ask your question..."
)


# ==================================================
# RAG PIPELINE
# ==================================================

if query:

    # ----------------------------------------------
    # USER MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",
            "content": query
        }

    )


    with st.chat_message("user"):

        st.write(query)


    # ----------------------------------------------
    # AI
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents... 🤔"
        ):

            try:

                # ==================================
                # STEP 1 — RETRIEVAL
                # ==================================

                docs = retrieve(

                    query,

                    top_k=10

                )


                # ==================================
                # STEP 2 — RERANKING
                # ==================================

                top_docs = rerank(

                    query,

                    docs,

                    top_k=5

                )


                # ==================================
                # STEP 3 — CONTEXT
                # ==================================

                context = "\n\n".join(
                    top_docs
                )


                # ==================================
                # STEP 4 — LLM
                # ==================================

                answer = generate_answer(

                    query,

                    context

                )


                # ==================================
                # SHOW ANSWER
                # ==================================

                st.write(answer)


                # ==================================
                # SAVE ANSWER
                # ==================================

                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": answer
                    }

                )


            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )