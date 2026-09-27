from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from backend.rag import process_video


app = FastAPI(
    title="YouTube RAG API",
    description="RAG-based question answering over YouTube transcripts",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# IN-MEMORY SESSIONS
# --------------------------------------------------

sessions = {}


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ProcessVideoRequest(BaseModel):
    video_url: str
    language: str
    api_key: str
    embedding_model: str
    chat_model: str


class AskQuestionRequest(BaseModel):
    session_id: str
    question: str


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.get("/")
def root():
    return FileResponse("frontend/index.html")


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

@app.post("/process-video")
def process_video_endpoint(
    request: ProcessVideoRequest
):

    try:

        rag_chain = process_video(
            video_url=request.video_url,
            language=request.language,
            api_key=request.api_key,
            embedding_model=request.embedding_model,
            chat_model=request.chat_model
        )

        session_id = str(uuid4())

        sessions[session_id] = rag_chain

        return {
            "success": True,
            "session_id": session_id,
            "message": "Video processed successfully."
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/ask")
def ask_question(
    request: AskQuestionRequest
):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    if request.session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found. "
                "Please process the video again."
            )
        )


    try:

        rag_chain = sessions[
            request.session_id
        ]

        answer = rag_chain.invoke(
            request.question
        )

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)