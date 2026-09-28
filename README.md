# YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) application that lets users ask questions about YouTube videos and receive answers grounded in their transcripts.

Built using **LangChain, Google Gemini, FAISS, FastAPI, and HTML/CSS/JavaScript**, this project demonstrates how to connect transcript retrieval, vector embeddings, similarity search, and an LLM into a complete web application.

## Features

* **YouTube Transcript Retrieval:** Fetches video transcripts using a third-party transcript endpoint.
* **Text Chunking:** Splits transcripts into manageable chunks using `RecursiveCharacterTextSplitter`.
* **Embedding Generation:** Converts text chunks into vector embeddings using Google Gemini embedding models.
* **Vector Storage:** Stores embeddings in a FAISS vector store.
* **Semantic Retrieval:** Retrieves the four most relevant transcript chunks for a question.
* **RAG-Based Question Answering:** Uses a Gemini chat model to generate answers from retrieved context.
* **Model Selection:** Allows users to select embedding and chat models.
* **Interactive Web Interface:** Enter a YouTube URL, choose a transcript language, process the video, and ask questions.
* **Session-Based Workflow:** Maintains the processed RAG chain in backend memory for follow-up questions.

## Tech Stack

| Technology            | Purpose                              |
| --------------------- | ------------------------------------ |
| Python                | Core application logic               |
| LangChain             | RAG pipeline and LCEL composition    |
| Google Gemini API     | Embeddings and answer generation     |
| FAISS                 | Vector storage and similarity search |
| FastAPI               | Backend API                          |
| HTML, CSS, JavaScript | Frontend interface                   |
| Uvicorn               | ASGI server                          |
| youtube-transcript.ai | Transcript retrieval endpoint        |

## How It Works

The application follows a standard Retrieval-Augmented Generation workflow.

### 1. Video Processing

1. The user provides a YouTube URL, Gemini API key, transcript language, embedding model, and chat model.
2. FastAPI receives the request through `/process-video`.
3. The application extracts the YouTube video ID.
4. The transcript is retrieved through the configured transcript endpoint.
5. `RecursiveCharacterTextSplitter` divides the transcript into chunks.
6. Gemini generates embeddings for the chunks.
7. FAISS stores the embeddings and associated documents.
8. A retriever and LangChain RAG chain are created.
9. The chain is stored in backend memory and associated with a session ID.

### 2. Question Answering

1. The user submits a question.
2. FastAPI receives the request through `/ask`.
3. The retriever searches FAISS for the four most relevant transcript chunks.
4. `RunnableParallel` prepares the retrieved context and original question.
5. `PromptTemplate` combines the context and question.
6. Gemini generates an answer using the supplied context.
7. `StrOutputParser` extracts the text response.
8. The answer is returned to the frontend.

## RAG Pipeline

```text
YouTube URL
    ↓
Extract Video ID
    ↓
Retrieve Transcript
    ↓
RecursiveCharacterTextSplitter
    ↓
Text Chunks
    ↓
Gemini Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search (Top 4 Chunks)
    ↓
RunnableParallel
    ├── Context → RunnableLambda → Format Documents
    └── Question → RunnablePassthrough
    ↓
PromptTemplate
    ↓
Gemini Chat Model
    ↓
StrOutputParser
    ↓
Final Answer
```

## Text Splitting Configuration

The project uses the following text-splitting configuration:

| Parameter                 |             Value |
| ------------------------- | ----------------: |
| Chunk size                |  1,000 characters |
| Chunk overlap             |    200 characters |
| Retrieved documents (`k`) |                 4 |
| Retrieval strategy        | Similarity search |

Chunk overlap preserves some continuity between neighboring chunks, while similarity search retrieves transcript sections relevant to the user's question.

## LangChain Concepts Implemented

This project helped me apply the following LangChain concepts:

* **Document creation:** Representing transcript chunks as documents.
* **Text splitters:** Preparing long transcripts for retrieval.
* **Embeddings:** Converting text into vector representations.
* **Vector stores:** Storing and searching embeddings using FAISS.
* **Retrievers:** Finding relevant documents for a query.
* **Prompt templates:** Structuring the context and question for the LLM.
* **LCEL (LangChain Expression Language):** Connecting components into a pipeline.
* **RunnableParallel:** Preparing context and question inputs in parallel.
* **RunnableLambda:** Formatting retrieved documents.
* **RunnablePassthrough:** Passing the original question through the chain.
* **StrOutputParser:** Returning the generated answer as text.

## 📁 Project Structure

```text
youtube-rag/
│
├── backend/
│   ├── main.py          # FastAPI endpoints and session handling
│   └── rag.py           # Transcript retrieval and RAG pipeline
│
├── frontend/
│   ├── index.html       # Application interface
│   ├── style.css        # Styling and responsive layout
│   └── script.js        # Frontend interactions and API calls
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Run Locally

### Prerequisites

* Python 3.11 or a compatible Python version
* A Google Gemini API key
* Git

### 1. Clone the repository

```bash
git clone https://github.com/ArpitJadhao/youtube-rag.git
cd youtube-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

Run the following command from the project root:

```bash
uvicorn backend.main:app --reload
```

### 5. Open the application

Visit:

http://127.0.0.1:8000

You can also access the automatically generated API documentation at:

http://127.0.0.1:8000/docs

## 🔑 API Key Handling

Users provide their own Gemini API key through the frontend.

* The frontend temporarily stores the key in browser `sessionStorage`.
* The key is sent to the backend when processing a video.
* The backend uses the key for embedding generation and chat model calls.
* The application does not intentionally persist the key in a database.

**Security note:** The backend retains the configured model/chain in memory, which may retain credentials in process memory. Use your own API key, avoid sharing it, and do not commit API keys to GitHub.

## ⚠️ Limitations

* Transcript availability depends on the video, selected language, and third-party transcript service.
* The application requires a valid Gemini API key and an accessible model.
* FAISS is created in memory for each processed video; the vector index is not permanently persisted.
* RAG sessions are stored in application memory and may be lost when the server restarts or the application instance changes.
* Free-tier API quotas, model availability, and service limits may affect usage.
* The application answers based on retrieved transcript context, so answers may be incomplete when relevant information is not retrieved.

## Learning Outcomes

Through this project, I gained practical experience with:

* Building a RAG pipeline from transcript ingestion to answer generation.
* Connecting embedding models, vector stores, retrievers, prompts, and LLMs.
* Using LCEL to compose reusable processing chains.
* Exposing a Python RAG pipeline through FastAPI.
* Connecting a JavaScript frontend to a Python backend.
* Managing user-provided API keys and temporary sessions.
* Deploying a machine-learning application as a web interface.

## Possible Future Improvements

* Add answer citations linked to retrieved transcript chunks.
* Evaluate retrieval and answer quality using RAG evaluation tools.
* Add chat history and multi-turn conversations.
* Improve transcript error handling and retrieval quality.
* Add persistent vector storage if needed.

These are potential future improvements and are not part of the current implementation.

## 👨‍💻 Author

**Arpit Jadhao**

GitHub: [ArpitJadhao](https://github.com/ArpitJadhao)

---

*Built as a hands-on project to understand and implement LangChain, vector retrieval, LCEL, and Retrieval-Augmented Generation in a working web application.*
