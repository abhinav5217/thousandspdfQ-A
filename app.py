import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# URL of the deployed FastAPI backend.
API_URL = "https://thousandspdfq-a.onrender.com"

# Configure the Streamlit page.
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

# Keep the conversation available during Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Track uploaded files in the current session.
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 PDF AI Assistant")

st.write(
    "Upload multiple PDFs and ask questions about their content."
)


# --------------------------------------------------
# CHECK BACKEND
# --------------------------------------------------

with st.sidebar:
    st.subheader("Backend Status")

    if st.button("Check API Status"):
        try:
            # Send a request to the deployed health endpoint.
            response = requests.get(
                f"{API_URL}/health",
                timeout=120
            )

            if response.ok:
                st.success("API is online!")
                st.json(response.json())
            else:
                st.error(f"API returned status {response.status_code}")

        except requests.RequestException as error:
            st.error(f"Could not connect to API: {error}")


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.subheader("📄 Upload PDFs")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Selected PDFs: {len(uploaded_files)}")

    for uploaded_file in uploaded_files:
        st.write(f"📄 {uploaded_file.name}")

    if st.button("🚀 Upload and Process PDFs"):
        try:
            # Prepare all selected PDFs for the FastAPI endpoint.
            files = [
                (
                    "files",
                    (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                )
                for uploaded_file in uploaded_files
            ]

            with st.spinner("Uploading PDFs to the backend..."):
                # Send PDFs to the deployed backend.
                response = requests.post(
                    f"{API_URL}/upload-multiple",
                    files=files,
                    timeout=300
                )

            # Display backend errors if the upload failed.
            if not response.ok:
                st.error(f"Upload failed: {response.text}")
            else:
                result = response.json()

                st.write("### Processing Results")
                st.json(result)

                # Record successfully processed files.
                for item in result.get("files", []):
                    if item.get("success"):
                        filename = item["filename"]

                        if filename not in st.session_state.processed_files:
                            st.session_state.processed_files.append(filename)

                if result.get("total_chunks_added", 0) > 0:
                    st.success("PDF processing completed!")

        except requests.RequestException as error:
            st.error(f"Could not upload PDFs: {error}")


# --------------------------------------------------
# PROCESSED FILES
# --------------------------------------------------

if st.session_state.processed_files:
    st.subheader("📚 Processed PDFs")

    for filename in st.session_state.processed_files:
        st.write(f"✅ {filename}")


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# --------------------------------------------------
# ASK A QUESTION
# --------------------------------------------------

question = st.chat_input("Ask a question about your PDFs...")

if question:
    # Show and save the user's message.
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # Ask the deployed backend to retrieve documents and generate an answer.
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=180
                )

            if not response.ok:
                st.error(f"Question failed: {response.text}")
            else:
                result = response.json()
                answer = result.get("answer", "No answer was returned.")

                st.write(answer)

                # Save the answer in the chat history.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

        except requests.RequestException as error:
            st.error(f"Could not contact the backend: {error}")


# --------------------------------------------------
# CLEAR CHAT
# --------------------------------------------------

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()