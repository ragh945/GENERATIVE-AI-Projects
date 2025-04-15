import streamlit as st
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

##Langsmith Tracking
os.environ["Langchain_project"]="Enhanced Open AI Q&A Chatbot"
os.environ["Langchain_api_key"]=os.getenv("Langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"]="true"

#Prompt Template
prompt=ChatPromptTemplate.from_messages([
    ("system","You are helpful AI assistant response to user queries"),
    ("user","{question}")
])

def generate_response(question,api_key,llm,temp,max_tokens):
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    response=chain.invoke({"question":question})
    return response

##Title of the APP
st.title("Enhanced Q&A ChatBot")

#Sidebar Settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your OPENAI API key",type="password")

##Dropdown to select Open ai models
llm=st.sidebar.selectbox("Select Your OpenAI models",["gpt-4o","DALLE","gpt-4","gpt-4-turbo"])

#Adjust response parameter
temp=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("max_tokens",min_value=50,max_value=300,value=150)

#User interface
st.write("🤖 **Go ahead and ask your question!**")

# Text input for user query
user_input = st.text_input("You:", placeholder="Type your question here...")

# Submit button with an emoji
if st.button("🚀 Submit"):
    if user_input:
        # Call your response generation function
        res = generate_response(user_input, api_key, llm, temp, max_tokens)
        st.write("💡 **Response:**")
        st.write(res)
    else:
        st.warning("⚠️ Please provide your query before submitting!")