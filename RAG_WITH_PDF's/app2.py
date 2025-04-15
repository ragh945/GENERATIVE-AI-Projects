# RAG Conversational Q&A Chatbot with message history
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("Conversational RAG with PDF Uploads and Chat history")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your GROQ API_KEY", type="password")

if api_key:
    llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")  # Correctly defining llm

    # Ensure session state for storing chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    session_id = st.text_input("Enter your session_id", value="default_session")

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for upload_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(upload_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        # Initialize embeddings & vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector = FAISS.from_documents(final_documents, embeddings)
        retriever = vector.as_retriever()

        # **Contextualized Question Prompt**
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood "
            "without a chat history. DO NOT answer the question, "
            "just reformulate it if needed and otherwise return as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # **Question Answering Prompt**
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # **User Input**
        user_input = st.text_input("Your Question ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)

else:
    st.write("Please Enter a Valid API Key")
