import urllib.request

from urllib.parse import urlparse, parse_qs

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import (
    StrOutputParser
)


# --------------------------------------------------
# EXTRACT YOUTUBE VIDEO ID
# --------------------------------------------------

def extract_video_id(url: str) -> str:

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname


    # youtube.com
    if hostname in [
        "youtube.com",
        "www.youtube.com"
    ]:

        query_params = parse_qs(
            parsed_url.query
        )

        # Normal YouTube URL
        if "v" in query_params:

            return query_params["v"][0]


        # YouTube Shorts
        if parsed_url.path.startswith(
            "/shorts/"
        ):

            return (
                parsed_url.path
                .split("/shorts/")[1]
                .split("/")[0]
            )


    # youtu.be
    if hostname in [
        "youtu.be",
        "www.youtu.be"
    ]:

        video_id = parsed_url.path.strip("/")

        if video_id:

            return video_id


    raise ValueError(
        "Invalid YouTube URL"
    )


# --------------------------------------------------
# GET YOUTUBE TRANSCRIPT
# --------------------------------------------------

def get_transcript(
    video_url: str,
    language: str = "en"
) -> str:

    video_id = extract_video_id(
        video_url
    )


    transcript_url = (
        f"https://youtube-transcript.ai/"
        f"transcript/{video_id}.txt"
        f"?lang={language}"
    )


    try:

        request = urllib.request.Request(

            transcript_url,

            headers={
                "User-Agent": "Mozilla/5.0"
            }

        )


        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            transcript = (
                response
                .read()
                .decode("utf-8")
            )


        if not transcript.strip():

            raise ValueError(
                "Transcript is empty."
            )


        return transcript


    except Exception as e:

        raise ValueError(
            f"Could not retrieve transcript: {str(e)}"
        )


# --------------------------------------------------
# CREATE VECTOR STORE
# --------------------------------------------------

def create_vector_store(
    transcript: str,
    api_key: str,
    embedding_model: str
):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = splitter.create_documents(
        [transcript]
    )


    embeddings = GoogleGenerativeAIEmbeddings(

        model=embedding_model,

        google_api_key=api_key

    )


    vector_store = FAISS.from_documents(

        chunks,

        embeddings

    )


    return vector_store


# --------------------------------------------------
# CREATE RAG CHAIN
# --------------------------------------------------

def create_rag_chain(
    vector_store,
    api_key: str,
    chat_model: str
):

    retriever = vector_store.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 4
        }

    )


    llm = ChatGoogleGenerativeAI(

        model=chat_model,

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


    # ----------------------------------------------
    # FORMAT RETRIEVED DOCUMENTS
    # ----------------------------------------------

    def format_docs(
        retrieved_docs
    ):

        return "\n\n".join(

            doc.page_content

            for doc in retrieved_docs

        )


    # ----------------------------------------------
    # PARALLEL CHAIN
    # ----------------------------------------------

    parallel_chain = RunnableParallel({

        "context": (

            retriever

            | RunnableLambda(
                format_docs
            )

        ),

        "question":
            RunnablePassthrough()

    })


    # ----------------------------------------------
    # OUTPUT PARSER
    # ----------------------------------------------

    parser = StrOutputParser()


    # ----------------------------------------------
    # MAIN RAG CHAIN
    # ----------------------------------------------

    main_chain = (

        parallel_chain

        | prompt

        | llm

        | parser

    )


    return main_chain


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

def process_video(
    video_url: str,
    language: str,
    api_key: str,
    embedding_model: str,
    chat_model: str
):

    # 1. Get transcript

    transcript = get_transcript(
        video_url,
        language
    )


    # 2. Create vector store

    vector_store = create_vector_store(

        transcript,

        api_key,

        embedding_model

    )


    # 3. Create RAG chain

    rag_chain = create_rag_chain(

        vector_store,

        api_key,

        chat_model

    )


    return rag_chain