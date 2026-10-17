import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
# URL of the existing deployed FastAPI backend.
API_URL = "https://thousandspdfq-a.onrender.com"

st.set_page_config(
    page_title="ThousandPDF | AI Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM UI THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark navy background with subtle indigo glow. */
    .stApp {
        background: radial-gradient(circle at 10% 0%, #172554 0, #0b1120 38%, #080d18 100%);
        color: #e5e7eb;
    }
    [data-testid="stHeader"] { background: rgba(8, 13, 24, .72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a30 0%, #0b1120 100%);
        border-right: 1px solid rgba(148, 163, 184, .16);
    }
    [data-testid="stSidebar"] * { color: #e5e7eb; }
    h1, h2, h3 { color: #f8fafc !important; letter-spacing: -.025em; }
    p, label, .stMarkdown { color: #cbd5e1; }
    .hero {
        padding: 1.8rem; border-radius: 22px; margin-bottom: 1.3rem;
        border: 1px solid rgba(129, 140, 248, .34);
        background: linear-gradient(120deg, rgba(49, 46, 129, .62), rgba(30, 41, 59, .72) 58%, rgba(14, 116, 144, .24));
        box-shadow: 0 18px 55px rgba(0, 0, 0, .2);
    }
    .hero-kicker { color: #a5b4fc; font-size: .76rem; font-weight: 800; letter-spacing: .16em; }
    .hero-title { color: #f8fafc; font-size: clamp(1.8rem, 4vw, 2.7rem); font-weight: 850; line-height: 1.15; margin: .5rem 0; }
    .hero-subtitle { color: #cbd5e1; max-width: 760px; font-size: 1rem; line-height: 1.7; }
    .pill { display: inline-block; padding: .38rem .72rem; margin: .7rem .35rem 0 0; border-radius: 999px; border: 1px solid rgba(165, 180, 252, .3); background: rgba(15, 23, 42, .48); color: #c7d2fe; font-size: .78rem; }
    [data-testid="stMetric"] { background: rgba(15, 23, 42, .72); border: 1px solid rgba(148, 163, 184, .17); padding: 1rem; border-radius: 16px; }
    [data-testid="stFileUploader"] { background: rgba(15, 23, 42, .55); border: 1px dashed rgba(129, 140, 248, .58); border-radius: 16px; padding: .6rem; }
    .stButton > button, .stFormSubmitButton > button {
        border: 1px solid rgba(129, 140, 248, .45); border-radius: 12px;
        font-weight: 700; min-height: 2.65rem; color: #fff;
        background: linear-gradient(100deg, #4f46e5, #2563eb);
        transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); color: #fff; }
    [data-testid="stChatMessage"] { background: rgba(15, 23, 42, .62); border: 1px solid rgba(148, 163, 184, .14); border-radius: 16px; padding: .8rem 1rem; }
    [data-testid="stChatInput"] textarea { background: #111b2e; border: 1px solid rgba(129, 140, 248, .36); border-radius: 14px; }
    hr { border-color: rgba(148, 163, 184, .18); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
# Preserve chat and uploaded-file names across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 ThousandPDF")
    st.caption("Your AI-powered document workspace")
    st.divider()
    st.markdown("### 💬 Conversation")
    st.metric("Messages in chat", len(st.session_state.messages))
    st.metric("PDFs processed this session", len(st.session_state.processed_files))
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Upload your documents, then ask questions about their content.")

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------
st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">AI DOCUMENT WORKSPACE</div>
      <div class="hero-title">Chat with your PDFs ✨</div>
      <div class="hero-subtitle">
        Turn long documents into clear answers. Upload one or more PDFs, retrieve
        relevant passages, and explore your documents through an AI conversation.
      </div>
      <span class="pill">⚡ RAG-powered Q&A</span>
      <span class="pill">📄 Multiple PDF upload</span>
      <span class="pill">💬 Conversational answers</span>
    </section>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# QUICK OVERVIEW
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 PDFs processed", len(st.session_state.processed_files))
with col2:
    st.metric("💬 Chat messages", len(st.session_state.messages))
with col3:
    st.metric("🧠 Experience", "Document Q&A")

# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------
st.markdown("## 📂 Build your document library")
st.write("Choose one or more PDF files to make their content available for questions.")

uploaded_files = st.file_uploader(
    "Drop PDFs here or browse files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple PDF files if you want to ask questions across documents.",
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} PDF(s) selected")
    with st.expander("Review selected files", expanded=True):
        for uploaded_file in uploaded_files:
            st.write(f"📄 **{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB")

    if st.button("✨ Upload and process PDFs", type="primary", use_container_width=True):
        try:
            # Match the existing FastAPI /upload-multiple endpoint format.
            files = [
                ("files", (item.name, item.getvalue(), "application/pdf"))
                for item in uploaded_files
            ]
            with st.spinner("Reading PDFs and preparing document chunks..."):
                response = requests.post(
                    f"{API_URL}/upload-multiple",
                    files=files,
                    timeout=300,
                )

            if not response.ok:
                st.error(f"Upload failed: {response.text}")
            else:
                result = response.json()
                successful = [item for item in result.get("files", []) if item.get("success")]
                failures = [item for item in result.get("files", []) if not item.get("success")]

                for item in successful:
                    filename = item.get("filename", "PDF")
                    if filename not in st.session_state.processed_files:
                        st.session_state.processed_files.append(filename)

                if successful:
                    st.success(
                        f"Processed {len(successful)} PDF(s) successfully. "
                        f"Chunks added: {result.get('total_chunks_added', 0)}."
                    )
                for item in failures:
                    st.warning(
                        f"{item.get('filename', 'PDF')}: "
                        f"{item.get('error', 'Processing failed')}"
                    )
        except requests.RequestException as error:
            st.error(f"Could not upload PDFs. Please try again. Details: {error}")

if st.session_state.processed_files:
    with st.expander("📚 Processed documents", expanded=False):
        for filename in st.session_state.processed_files:
            st.markdown(f"- ✅ {filename}")

st.divider()

# --------------------------------------------------
# CHAT WITH DOCUMENTS
# --------------------------------------------------
st.markdown("## 💬 Ask your documents")
st.caption("Try a question, request a summary, or ask for the key points from your PDFs.")

# Quick prompts help users start a conversation without typing a full question.
prompt_cols = st.columns(3)
for col, prompt in zip(
    prompt_cols,
    ["Summarize the main ideas", "Explain the key concepts", "What are the important points?"],
):
    with col:
        if st.button(prompt, use_container_width=True):
            st.session_state["suggested_question"] = prompt

# Render the existing conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDFs...")
if not question and st.session_state.get("suggested_question"):
    question = st.session_state.pop("suggested_question")

if question:
    # Display and save the user's question.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Keep the existing API contract so the deployed backend remains compatible.
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents and preparing an answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=180,
                )

            if not response.ok:
                st.error(f"Question failed: {response.text}")
            else:
                result = response.json()
                answer = result.get("answer", "No answer was returned.")
                st.markdown(answer)
                if result.get("documents_retrieved") is not None:
                    st.caption(
                        f"Retrieved {result.get('documents_retrieved')} chunks · "
                        f"Used {result.get('documents_used', '—')} in the answer"
                    )
                # Save the assistant answer for the next rerun.
                st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.RequestException as error:
            st.error(f"Could not contact the document service. Details: {error}")

# Footer intentionally has no backend-status panel or health-check button.
st.divider()
st.caption("Built for focused reading · ThousandPDF AI")
