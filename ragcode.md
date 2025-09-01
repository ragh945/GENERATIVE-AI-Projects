```bash
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# 1. Load sample documents
docs = [
    Document(page_content="Kubernetes is an open-source platform for container orchestration."),
    Document(page_content="Docker is used to containerize applications."),
    Document(page_content="Terraform is used for infrastructure as code to provision cloud resources."),
]

# 2. Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# 3. Create embeddings & vector database
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Create retriever + LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# 5. Ask a question
query = "What is Kubernetes used for?"
result = qa_chain.run(query)

print("Q:", query)
print("A:", result)
```
