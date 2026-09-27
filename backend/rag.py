from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Extract YouTube Video ID
# --------------------------------------------------

def extract_video_id(url: str) -> str:
    """
    Extract the video ID from common YouTube URL formats.
    """

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    # youtube.com/watch?v=VIDEO_ID
    if hostname in ["youtube.com", "www.youtube.com"]:
        query_params = parse_qs(parsed_url.query)

        if "v" in query_params:
            return query_params["v"][0]

        # youtube.com/shorts/VIDEO_ID
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1].split("/")[0]

    # youtu.be/VIDEO_ID
    if hostname in ["youtu.be", "www.youtu.be"]:
        video_id = parsed_url.path.strip("/")

        if video_id:
            return video_id

    raise ValueError("Invalid YouTube URL")


# --------------------------------------------------
# 2. Get YouTube Transcript
# --------------------------------------------------

def get_transcript(
    video_url: str,
    language: str = "en"
) -> str:

    video_id = extract_video_id(video_url)

    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id,
            languages=[language]
        )

        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        if not transcript.strip():
            raise ValueError("Transcript is empty.")

        return transcript

    except TranscriptsDisabled:
        raise ValueError(
            "This video does not have captions enabled."
        )

    except NoTranscriptFound:
        raise ValueError(
            f"No transcript was found for language '{language}'."
        )

    except VideoUnavailable:
        raise ValueError(
            "This YouTube video is unavailable."
        )


# --------------------------------------------------
# 3. Create FAISS Vector Store
# --------------------------------------------------

def create_vector_store(
    transcript: str,
    api_key: str
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=api_key
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# --------------------------------------------------
# 4. Create RAG Chain
# --------------------------------------------------

def create_rag_chain(
    vector_store,
    api_key: str
):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    def format_docs(retrieved_docs):

        return "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

    # Retrieval + formatting
    parallel_chain = RunnableParallel({
        "context": (
            retriever
            | RunnableLambda(format_docs)
        ),

        "question": RunnablePassthrough()
    })

    # Output parser
    parser = StrOutputParser()

    # Complete RAG chain
    main_chain = (
        parallel_chain
        | prompt
        | llm
        | parser
    )

    return main_chain


# --------------------------------------------------
# 5. Process YouTube Video
# --------------------------------------------------

def process_video(
    video_url: str,
    language: str,
    api_key: str
):

    # Step 1: Get transcript
    transcript = get_transcript(
        video_url,
        language
    )

    # Step 2: Create vector store
    vector_store = create_vector_store(
        transcript,
        api_key
    )

    # Step 3: Create RAG chain
    rag_chain = create_rag_chain(
        vector_store,
        api_key
    )

    return rag_chain