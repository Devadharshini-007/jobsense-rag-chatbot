# 🧠 JobSense

**An AI-powered chatbot that helps you understand any job, check your fit, and prepare to apply — all through natural conversation.**

Built as a RAG (Retrieval-Augmented Generation) application using LangChain-style architecture, FAISS vector search, and Groq's LLM API.

---

## What it does

Attach a job description and your resume, and JobSense becomes your personal job-application assistant:

- **📄 Understand any JD** — ask questions about requirements, responsibilities, or anything in the job posting
- **🎯 Check your fit** — get an honest fit score with specific strengths and gaps
- **✍️ Tailor your resume** — get a complete, ATS-friendly rewritten resume targeting that specific job
- **💌 Generate a cover letter** — a genuine, specific cover letter using your real experience
- **🎤 Prepare for the interview** — likely technical and behavioral questions, tailored to your background and the role
- **⬇️ Export anything as a Word doc** — just ask, and download a polished `.docx`
- **💬 All through one chat** — no forms, no steps, just talk to it like ChatGPT or Claude

---

## How it works

```
Job Description / Resume (PDF or text)
            ↓
    Text Chunking + Embedding (Sentence Transformers)
            ↓
        FAISS Vector Index
            ↓
User Question → Semantic Search → Relevant Context
            ↓
    Groq LLM (openai/gpt-oss-20b) + Context
            ↓
        Grounded, Specific Answer
```

The core RAG pipeline retrieves only the most relevant sections of the job description for each question, instead of stuffing the entire document into every prompt — keeping answers accurate and reducing hallucination.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| Backend Logic | Python |
| Frontend | Streamlit |
| PDF Parsing | pypdf |
| Document Export | python-docx |
| Conversation Persistence | SQLite |

---

## Features in Detail

### Smart Document Detection
Attach your resume and JD in any order — JobSense automatically figures out which is which using filename hints and content analysis, no manual labeling needed.

### Conversation History
Every conversation is automatically saved locally. Pick up right where you left off, even after closing the app.

### Export to Word
Ask "give me this as a docx" after any fit analysis, tailored resume, cover letter, or interview prep — get a clean, properly formatted Word document.

---

## Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/Devadharshini-007/jobsense-rag-chatbot
cd jobsense-rag-chatbot
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## Project Structure

```
jobsense-rag-chatbot/
├── app/
│   ├── embeddings.py          # Text chunking + FAISS vector search
│   ├── rag_pipeline.py        # RAG logic, fit scoring, content generation
│   ├── pdf_utils.py           # PDF text extraction
│   ├── docx_export.py         # Word document generation
│   └── conversation_store.py  # SQLite-based chat history persistence
├── streamlit_app.py           # Main chat application
└── requirements.txt
```

---

## Why I Built This

As a fresher applying to AI/ML and Data Analyst roles, I was tired of manually cross-referencing job descriptions against my resume for every application. JobSense automates that — and doubles as a hands-on project demonstrating RAG architecture, vector search, LLM prompt engineering, and full-stack Python development.

---

## Future Improvements

- Multi-language support
- Browser extension for one-click JD capture from job boards
- Comparison mode across multiple job postings
- Deployed live version (Streamlit Cloud)

---

## License

This project is for educational and portfolio purposes.
