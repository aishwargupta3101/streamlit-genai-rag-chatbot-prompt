import streamlit as st
import os
import pytesseract
from PIL import Image
from dotenv  import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders  import(
        PyPDFLoader,
        TextLoader,
        CSVLoader,
        Docx2txtLoader
)
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq


load_dotenv()

st.title(" RAG CHATBOT")
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages=[]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



if not os.path.exists("temp"):
    os.makedirs("temp")

uploaded_file= st.file_uploader(
    "upload",
    type=["pdf", "docx","txt", "csv", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

documents=[]
if uploaded_file:
    with st.spinner(" file processing"):
        for file in uploaded_file:  # type:ignore
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

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
                docs = Document(page_content=text)
                documents.append(docs)
        st.success("file processed successfully")



text_splitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
chunks=[]
vector_db= None
if documents:
    chunks = text_splitter.split_documents(documents)
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
embedding= load_embedding()

if chunks:
    st.session_state.vector_db= Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="vector_db",

    )
elif os.path.exists("vector_db"):
    st.session_state.vector_db= Chroma(
        persist_directory="vector_db",
        embedding_function=embedding
    )




llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt_template=PromptTemplate(
    input_variables=["context", "question"],
    template=
    """
    You are a peaceful assistant.
    Use the context to answer the question accurately.
    Rules:
    - If the answer exists in the context, use it.
    - If the context is insufficient, use your general knowledge.
    - Answer clearly and concisely.
     
    Context:
    {context}
    Question:
    {question}
    
    Answer the question clearly..:
    """
    
)

query= st.chat_input("Ask the question")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").write(query)
    if st.session_state.vector_db:
        docs= st.session_state.vector_db.similarity_search(query, k=3)
        context="\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source", "unknown") for doc in docs]
    else:
        context= "No documents"
        sources=[]


    prompt=prompt_template.format(
        context=context,
        question= query
    )

    response= llm.invoke(prompt)
    st.chat_message("assistant").write(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    if sources:
        st.markdown("**Sources:**")
        for s in set(sources):
            st.write(s)











