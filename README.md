# 🤖 Generative AI Projects Collection

This repository contains a curated set of hands-on **Generative AI projects** designed using Large Language Models (LLMs), RAG (Retrieval-Augmented Generation), LangChain, and various open-source tools. These projects demonstrate how to apply GenAI to real-world use cases like chatbots, voice bots, database querying, and intelligent evaluations.

---

## 📌 What is Generative AI?

**Generative AI** refers to AI models capable of generating human-like content — such as text, images, audio, or code — by learning from massive datasets. These models can perform tasks like:
- Text summarization
- Answer generation
- Code completion
- Chat conversations
- SQL generation from natural language

---

## 🧠 What are Large Language Models (LLMs)?

**LLMs** are deep learning models trained on extensive text corpora to understand, generate, and manipulate human language. Examples include:
- GPT (OpenAI)
- LLaMA (Meta)
- Mistral
- Gemma
- Falcon
- Claude

LLMs power GenAI systems in tasks like Q&A, writing, translation, code generation, and more.

---

## 📂 Project Highlights

### 🔎 1. Retrieval-Augmented Generation (RAG)
- Combines LLMs with external data sources to provide accurate, grounded responses.
- Uses `FAISS` for vector similarity search and `HuggingFace` embeddings.
- Loads knowledge from:
  - PDFs (via `PyPDFLoader`)
  - Web URLs (via `WebBaseLoader`)
- Applications: Chatbots with memory, document Q&A, knowledge agents.

---

### 💬 2. AI Chatbot with RAG
- Streamlit-based chatbot using LLM + RAG pipeline.
- Fetches context-aware answers from pre-loaded documents or websites.
- Supports multiple domains like Python, SQL, ML, DL, Power BI, GenAI.
- Easily extendable for enterprise FAQs or customer support.

---

### 🎙️ 3. AI Voice Interview Chatbot
- Voice-interactive mock interview assistant.
- User selects role (Data Scientist / Analyst) and topics (Python, SQL, etc.).
- Features:
  - Voice greeting and instructions
  - 5 random questions (easy → hard)
  - Accepts voice/text answers (10 sec limit for voice)
  - Evaluates with LLMs + keyword/similarity scoring
  - Final report with score, strengths, improvements, and bar chart visualization
  - Speaks feedback aloud at the end
- Uses `gTTS`, `speech_recognition`, `pygame`, Streamlit.

---

### 🧾 4. SQL with LLMs – Natural Language to SQL
- Translates user questions into executable SQL queries using LLMs.
- Connects to real databases and fetches actual results.
- Example: “Show top 10 customers by revenue in 2023.”
- Ideal for non-technical business users.

---

## 📘 Key GenAI Topics Covered
- ✅ Large Language Models (LLMs)
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Semantic Search & Embeddings
- ✅ Prompt Engineering
- ✅ Text-to-SQL with LLMs
- ✅ Natural Language Understanding
- ✅ Voice Interaction using TTS and STT
- ✅ Evaluation using Similarity and Keyword Matching
- ✅ Streamlit Deployment for AI apps

---

## 🧰 Libraries & Frameworks Used

| Category | Libraries / Tools |
|---------|--------------------|
| LLMs & Embeddings | `transformers`, `sentence-transformers`, `Groq`, `HuggingFace` |
| RAG & Pipelines | `langchain`, `faiss-cpu`, `PyPDFLoader`, `WebBaseLoader` |
| Voice | `speech_recognition`, `gTTS`, `pygame`, `audio_recorder_streamlit` |
| Web UI | `streamlit`, `matplotlib` |
| Evaluation | `scikit-learn`, `numpy`, `difflib`, `textdistance` |
| Data | `pandas`, `sqlalchemy` |
| Visualization | `matplotlib`, `seaborn` |

---

## 🧪 How to Run the Projects

Each folder contains:
- 📄 README with setup instructions
- 🧠 Model & Data loading logic
- 🚀 Streamlit app file (`app.py`)
- 📁 `requirements.txt`

Clone the repo and follow individual instructions to explore each project.

```bash
git clone https://github.com/your-username/genai-projects.git
cd genai-projects/<project-name>
pip install -r requirements.txt
streamlit run app.py
