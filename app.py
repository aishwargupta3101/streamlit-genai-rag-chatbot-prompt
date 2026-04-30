import streamlit as st
import os
import uuid
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from gtts import gTTS
import tempfile
import whisper
from audiorecorder import audiorecorder

load_dotenv()
st.set_page_config(page_title="RAG Voice Chatbot",layout="wide")
st.title("🤖 RAG Chatbot")

enable_voice = st.checkbox("🔊 Voice Response", value=True)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = whisper.load_model("base")
whisper_model= st.session_state.whisper_model

def speak_text(text):
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name,format="audio/mp3")
        os.remove(fp.name)
    except Exception as e:
        st.warning(f"TTS Error: {e}")

if not os.path.exists("temp"):
    os.makedirs("temp")
uploaded_file = st.file_uploader(
    "Upload files",
    type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)
documents = []

if uploaded_file:
    with st.spinner("Processing files..."):
        for file in uploaded_file:
            file_path = os.path.join("temp", file.name)

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            try:

                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.name.endswith(".csv"):
                    loader = CSVLoader(file_path)
                    documents.extend(loader.load())

                elif file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                elif file.name.endswith((".png", "jpg", "jpeg")):
                    image = Image.open(file_path)
                    text = pytesseract.image_to_string(image)
                    documents.append(Document(page_content=text))
            except Exception as e:
                st.error(f"error processing {file.name}: {e}")
        st.success("Files processed successfully!")
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
chunks = []
if documents:
    chunks = text_splitter.split_documents(documents)

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = load_embedding()

if chunks and st.session_state.vector_db is None:
    st.session_state.vector_db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="vector_db"
    )
elif os.path.exists("vector_db") and st.session_state.vector_db is None:
    st.session_state.vector_db = Chroma(
        persist_directory="vector_db",
        embedding_function=embedding
    )

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant.
    Use context if available, otherwise general knowledge.
    Keep answer clear and concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

st.subheader("🎤 Speak your question")
audio = audiorecorder("Start Recording", "Stop Recording")

query = None
if len(audio) > 0:
    audio_path = f"temp/{uuid.uuid4()}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio.export().read())
    try:
        with st.spinner("Transcribing..."):
            result = whisper_model.transcribe("input.wav")
            query = result["text"]
        st.chat_message("user").write(f"🗣 {query}")
        st.session_state.messages.append({"role": "user", "content": query})
    except Exception as e:
        st.error(f"Transcription Error :{e}")

text_query = st.chat_input("💬 Type your question...")

if text_query:
    query = text_query
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

# ------------------ RESPONSE ------------------
if query:
    with st.spinner(" Thinking....."):

        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            context = "No documents available"
        prompt = prompt_template.format(
            context=context,
            question=query
        )
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response,"content")else str(response)
        except Exception as e:
            response_text= f"Error generating response: {e}"

    st.chat_message("assistant").write(response.content)

    if enable_voice:
        speak_text(response.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })
if st.button(" Clear Chat"):
    st.session_state.messages = []
