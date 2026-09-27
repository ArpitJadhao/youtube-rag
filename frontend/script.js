const apiKeyInput = document.getElementById("apiKey");
const toggleKeyButton = document.getElementById("toggleKey");

const videoUrlInput = document.getElementById("videoUrl");
const languageSelect = document.getElementById("language");

const embeddingModelSelect =
    document.getElementById("embeddingModel");

const chatModelSelect =
    document.getElementById("chatModel");

const processButton = document.getElementById("processButton");
const processStatus = document.getElementById("processStatus");

const questionInput = document.getElementById("question");
const askButton = document.getElementById("askButton");

const answer = document.getElementById("answer");
const errorMessage = document.getElementById("errorMessage");


// --------------------------------------------------
// SESSION STORAGE
// --------------------------------------------------

const savedApiKey =
    sessionStorage.getItem("gemini_api_key");

const savedSessionId =
    sessionStorage.getItem("rag_session_id");

if (savedApiKey) {
    apiKeyInput.value = savedApiKey;
}

let sessionId = savedSessionId || null;


// --------------------------------------------------
// SHOW / HIDE API KEY
// --------------------------------------------------

toggleKeyButton.addEventListener("click", () => {

    if (apiKeyInput.type === "password") {

        apiKeyInput.type = "text";
        toggleKeyButton.textContent = "Hide";

    } else {

        apiKeyInput.type = "password";
        toggleKeyButton.textContent = "Show";

    }

});


// --------------------------------------------------
// SAVE API KEY FOR CURRENT SESSION
// --------------------------------------------------

apiKeyInput.addEventListener("input", () => {

    const apiKey = apiKeyInput.value.trim();

    if (apiKey) {

        sessionStorage.setItem(
            "gemini_api_key",
            apiKey
        );

    } else {

        sessionStorage.removeItem(
            "gemini_api_key"
        );

    }

});


// --------------------------------------------------
// PROCESS VIDEO
// --------------------------------------------------

processButton.addEventListener("click", async () => {

    clearError();

    const apiKey = apiKeyInput.value.trim();
    const videoUrl = videoUrlInput.value.trim();
    const language = languageSelect.value;

    const embeddingModel =
        embeddingModelSelect.value;

    const chatModel =
        chatModelSelect.value;


    // ----------------------------------------------
    // VALIDATION
    // ----------------------------------------------

    if (!apiKey) {

        showError(
            "Please enter your Gemini API key."
        );

        return;
    }

    if (!videoUrl) {

        showError(
            "Please enter a YouTube URL."
        );

        return;
    }


    // ----------------------------------------------
    // SAVE API KEY
    // ----------------------------------------------

    sessionStorage.setItem(
        "gemini_api_key",
        apiKey
    );


    // ----------------------------------------------
    // UI STATE
    // ----------------------------------------------

    processButton.disabled = true;
    processButton.textContent = "Processing...";

    processStatus.textContent =
        "Fetching transcript and creating the RAG index...";

    askButton.disabled = true;

    sessionId = null;


    // ----------------------------------------------
    // PROCESS VIDEO
    // ----------------------------------------------

    try {

        const response = await fetch(
            "/process-video",
            {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    video_url: videoUrl,

                    language: language,

                    api_key: apiKey,

                    embedding_model:
                        embeddingModel,

                    chat_model:
                        chatModel

                })

            }
        );


        const data = await response.json();


        // ------------------------------------------
        // HANDLE ERROR
        // ------------------------------------------

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Failed to process the video."
            );

        }


        // ------------------------------------------
        // SAVE SESSION
        // ------------------------------------------

        sessionId = data.session_id;

        sessionStorage.setItem(
            "rag_session_id",
            sessionId
        );


        // ------------------------------------------
        // SUCCESS
        // ------------------------------------------

        processStatus.textContent =
            "Video processed successfully. You can now ask questions.";

        answer.textContent =
            "Video is ready. Ask a question about it.";

        askButton.disabled = false;


    } catch (error) {

        showError(error.message);

        processStatus.textContent =
            "Unable to process the video.";

    } finally {

        processButton.disabled = false;

        processButton.textContent =
            "Process Video";

    }

});


// --------------------------------------------------
// ASK QUESTION
// --------------------------------------------------

askButton.addEventListener("click", async () => {

    clearError();

    const question =
        questionInput.value.trim();


    // ----------------------------------------------
    // VALIDATION
    // ----------------------------------------------

    if (!sessionId) {

        showError(
            "Please process a video first."
        );

        return;
    }

    if (!question) {

        showError(
            "Please enter a question."
        );

        return;
    }


    // ----------------------------------------------
    // UI STATE
    // ----------------------------------------------

    askButton.disabled = true;

    askButton.textContent =
        "Thinking...";

    answer.textContent =
        "Searching the transcript...";


    // ----------------------------------------------
    // ASK BACKEND
    // ----------------------------------------------

    try {

        const response = await fetch(
            "/ask",
            {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    session_id: sessionId,

                    question: question

                })

            }
        );


        const data = await response.json();


        // ------------------------------------------
        // HANDLE ERROR
        // ------------------------------------------

        if (!response.ok) {

            // Session may have disappeared
            // after a server restart.

            if (response.status === 404) {

                sessionId = null;

                sessionStorage.removeItem(
                    "rag_session_id"
                );

                throw new Error(
                    "Your video session has expired. Please process the video again."
                );

            }

            throw new Error(
                data.detail ||
                "Failed to get an answer."
            );

        }


        // ------------------------------------------
        // DISPLAY ANSWER
        // ------------------------------------------

        answer.textContent =
            data.answer;


    } catch (error) {

        showError(
            error.message
        );

        answer.textContent =
            "Unable to generate an answer.";

    } finally {

        askButton.disabled = false;

        askButton.textContent =
            "Ask Question";

    }

});


// --------------------------------------------------
// ENTER KEY SUPPORT
// --------------------------------------------------

questionInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            if (!askButton.disabled) {

                askButton.click();

            }

        }

    }
);


// --------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------

function showError(message) {

    errorMessage.textContent =
        message;

}


function clearError() {

    errorMessage.textContent =
        "";

}