import streamlit as st
import requests
from langserve import RemoteRunnable

BACKEND_URL = "http://127.0.0.1:8000"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

st.set_page_config(page_title="Company Policy RAG Chatbot", layout="wide")
st.title("🤖 Company Policy RAG Chatbot")
st.write("Powered by LangServe, Groq, and FAISS")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Upload your policy documents (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Files") and uploaded_files:
        with st.spinner("Processing documents..."):
            
            files_to_upload = []
            for f in uploaded_files:
                files_to_upload.append(
                    ('files', (f.name, f.getvalue(), f.type))
                )
            
            try:
                response = requests.post(UPLOAD_ENDPOINT, files=files_to_upload)
                
                if response.status_code == 200:
                    st.success(response.json().get("message", "Files processed successfully!"))
                    st.session_state.files_processed = True
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    st.session_state.files_processed = False
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend server. Is it running?")
                st.session_state.files_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.info(f"Sources: {', '.join(message['sources'])}")

if prompt := st.chat_input("What is your question?"):
    
    if not st.session_state.files_processed:
        st.warning("Please upload and process your policy documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_chain = RemoteRunnable(CHAT_ENDPOINT)
                    
                    input_data = {"question": prompt}
                    
                    response = chat_chain.invoke(input_data)
                    
                    answer = response.get("answer", "No answer found.")
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.info(f"Sources: {', '.join(sources)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error communicating with backend: {e}")