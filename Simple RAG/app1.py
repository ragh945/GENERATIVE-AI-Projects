import os
#import sentence_transformers
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

st.title("🔍 AI-Powered Research Assistant (RAG) 📄")
load_dotenv()

# Load environment variables
os.environ["Groq_api_key"] = os.getenv("Groq_api_key")
os.environ["HF_token"] = os.getenv("HF_token")
groq_api_key = os.getenv("Groq_api_key")

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    """Answer the following questions based on the context.
    Please provide an accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vector" not in st.session_state:  # Changed from "vectors" to "vector"
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("Documents")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector database initialized successfully!")

# User input field
user_input = st.text_input("Enter your query from the research paper")

# Button to initialize vector database
if st.sidebar.button("Initialize Vector DB"):
    create_vector_embedding()

# Ensure vector is initialized before retrieval
if "vector" in st.session_state and st.session_state.vector is not None and user_input:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retrieval, document_chain)

    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")

    # Display response
    st.write(response["answer"])

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, docs in enumerate(response["context"]):
            st.write(docs.page_content)
            st.write("---------------------------------")

