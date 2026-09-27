// ============================================================
// GLOBAL STATE
// ============================================================

let totalAnalysis = 0;
let cyberbullyingCount = 0;
let safeCount = 0;
let totalConfidence = 0;

let detectionHistory = [];

let lastPrediction = null;


// ============================================================
// SESSION STORAGE
// ============================================================

const SESSION_STORAGE_KEY =
    "cyberbullying_detection_session";


// ============================================================
// DOM ELEMENTS
// ============================================================

const messageInput =
    document.getElementById("messageInput");

const analyzeButton =
    document.getElementById("analyzeButton");

const clearButton =
    document.getElementById("clearButton");

const characterCount =
    document.getElementById("characterCount");

const loadingState =
    document.getElementById("loadingState");

const errorMessage =
    document.getElementById("errorMessage");

const apiStatus =
    document.getElementById("apiStatus");

const resultStatusBadge =
    document.getElementById("resultStatusBadge");

const resultMain =
    document.getElementById("resultMain");

const resultDetails =
    document.getElementById("resultDetails");

const summaryText =
    document.getElementById("summaryText");

const predictionStatus =
    document.getElementById("predictionStatus");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceFill =
    document.getElementById("confidenceFill");

const severityValue =
    document.getElementById("severityValue");

const targetValue =
    document.getElementById("targetValue");

const categoryContainer =
    document.getElementById("categoryContainer");

const reasonText =
    document.getElementById("reasonText");

const severityReason =
    document.getElementById("severityReason");

const targetReason =
    document.getElementById("targetReason");

const suggestionText =
    document.getElementById("suggestionText");

const historyContainer =
    document.getElementById("historyContainer");

const clearHistoryButton =
    document.getElementById("clearHistoryButton");


// ============================================================
// SAVE SESSION DATA
// ============================================================

function saveSessionData() {

    const sessionData = {

        totalAnalysis:
            totalAnalysis,

        cyberbullyingCount:
            cyberbullyingCount,

        safeCount:
            safeCount,

        totalConfidence:
            totalConfidence,

        detectionHistory:
            detectionHistory,

        lastPrediction:
            lastPrediction

    };


    try {

        sessionStorage.setItem(
            SESSION_STORAGE_KEY,
            JSON.stringify(sessionData)
        );

    } catch (error) {

        console.error(
            "Failed to save session data:",
            error
        );

    }

}


// ============================================================
// LOAD SESSION DATA
// ============================================================

function loadSessionData() {

    const savedData =
        sessionStorage.getItem(
            SESSION_STORAGE_KEY
        );


    if (!savedData) {

        return;

    }


    try {

        const sessionData =
            JSON.parse(savedData);


        totalAnalysis =
            Number(
                sessionData.totalAnalysis || 0
            );


        cyberbullyingCount =
            Number(
                sessionData.cyberbullyingCount || 0
            );


        safeCount =
            Number(
                sessionData.safeCount || 0
            );


        totalConfidence =
            Number(
                sessionData.totalConfidence || 0
            );


        detectionHistory =
            Array.isArray(
                sessionData.detectionHistory
            )
                ? sessionData.detectionHistory.map(
                    (item) => ({

                        ...item,

                        timestamp:
                            new Date(
                                item.timestamp
                            )

                    })
                )
                : [];


        lastPrediction =
            sessionData.lastPrediction || null;


    } catch (error) {

        console.error(
            "Failed to load session data:",
            error
        );


        sessionStorage.removeItem(
            SESSION_STORAGE_KEY
        );

    }

}


// ============================================================
// RESTORE SESSION UI
// ============================================================

function restoreSessionUI() {

    // --------------------------------------------------------
    // MAIN STATISTICS
    // --------------------------------------------------------

    document.getElementById(
        "totalAnalysis"
    ).textContent =
        totalAnalysis;


    document.getElementById(
        "cyberbullyingCount"
    ).textContent =
        cyberbullyingCount;


    document.getElementById(
        "safeCount"
    ).textContent =
        safeCount;


    const averageConfidence =
        totalAnalysis > 0
            ? totalConfidence / totalAnalysis
            : 0;


    document.getElementById(
        "averageConfidence"
    ).textContent =
        `${Math.round(
            averageConfidence * 100
        )}%`;


    // --------------------------------------------------------
    // ANALYTICS
    // --------------------------------------------------------

    document.getElementById(
        "analyticsTotal"
    ).textContent =
        totalAnalysis;


    document.getElementById(
        "analyticsDetected"
    ).textContent =
        cyberbullyingCount;


    document.getElementById(
        "analyticsSafe"
    ).textContent =
        safeCount;


    const detectionRate =
        totalAnalysis > 0
            ? (
                cyberbullyingCount /
                totalAnalysis
            ) * 100
            : 0;


    document.getElementById(
        "detectionRate"
    ).textContent =
        `${Math.round(
            detectionRate
        )}%`;


    // --------------------------------------------------------
    // HISTORY
    // --------------------------------------------------------

    renderHistory();


    // --------------------------------------------------------
    // LAST PREDICTION
    // --------------------------------------------------------

    if (lastPrediction) {

        if (
            lastPrediction.text
        ) {

            messageInput.value =
                lastPrediction.text;

            characterCount.textContent =
                `${lastPrediction.text.length} / 1000`;

        }


        if (
            lastPrediction.data
        ) {

            displayPrediction(
                lastPrediction.data
            );

        }

    }

}


// ============================================================
// CHARACTER COUNTER
// ============================================================

messageInput.addEventListener(
    "input",
    () => {

        const length =
            messageInput.value.length;

        characterCount.textContent =
            `${length} / 1000`;

    }
);


// ============================================================
// CLEAR INPUT
// ============================================================

clearButton.addEventListener(
    "click",
    () => {

        messageInput.value =
            "";

        characterCount.textContent =
            "0 / 1000";

        messageInput.focus();

        hideError();

    }
);


// ============================================================
// EXAMPLE BUTTONS
// ============================================================

const exampleButtons =
    document.querySelectorAll(
        ".example-button"
    );


exampleButtons.forEach(
    (button) => {

        button.addEventListener(
            "click",
            () => {

                const text =
                    button.dataset.text;


                messageInput.value =
                    text;


                characterCount.textContent =
                    `${text.length} / 1000`;


                hideError();


                messageInput.focus();

            }
        );

    }
);


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeButton.addEventListener(
    "click",
    analyzeMessage
);


// ============================================================
// ENTER KEY SUPPORT
// ============================================================

messageInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            (event.ctrlKey || event.metaKey)
        ) {

            event.preventDefault();

            analyzeMessage();

        }

    }
);


// ============================================================
// ANALYZE MESSAGE
// ============================================================

async function analyzeMessage() {

    const text =
        messageInput.value.trim();


    hideError();


    // --------------------------------------------------------
    // VALIDATION
    // --------------------------------------------------------

    if (!text) {

        showError(
            "Please enter a message before analyzing."
        );

        messageInput.focus();

        return;

    }


    if (text.length > 1000) {

        showError(
            "Message cannot exceed 1000 characters."
        );

        return;

    }


    // --------------------------------------------------------
    // LOADING STATE
    // --------------------------------------------------------

    setLoading(true);


    try {

        const response =
            await fetch(
                "/predict",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })

                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );

        }


        // ----------------------------------------------------
        // SAVE LAST PREDICTION
        // ----------------------------------------------------

        lastPrediction = {

            text: text,

            data: data

        };


        // ----------------------------------------------------
        // DISPLAY RESULT
        // ----------------------------------------------------

        displayPrediction(
            data
        );


        // ----------------------------------------------------
        // UPDATE STATISTICS
        // ----------------------------------------------------

        updateStatistics(
            data
        );


        // ----------------------------------------------------
        // ADD HISTORY
        // ----------------------------------------------------

        addToHistory(
            text,
            data
        );


        // ----------------------------------------------------
        // SAVE SESSION
        // ----------------------------------------------------

        saveSessionData();


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        showError(
            error.message ||
            "Unable to connect to the prediction server."
        );


    } finally {

        setLoading(
            false
        );

    }

}


// ============================================================
// DISPLAY PREDICTION
// ============================================================

function displayPrediction(data) {

    const isCyberbullying =
        Boolean(
            data.cyberbullying
        );


    // --------------------------------------------------------
    // SHOW RESULT SECTION
    // --------------------------------------------------------

    resultMain.classList.add(
        "hidden"
    );

    resultDetails.classList.remove(
        "hidden"
    );


    // --------------------------------------------------------
    // STATUS
    // --------------------------------------------------------

    if (isCyberbullying) {

        resultStatusBadge.textContent =
            "Cyberbullying Detected";


        resultStatusBadge.className =
            "result-badge danger";


        predictionStatus.textContent =
            "Detected";


        predictionStatus.className =
            "metric-value danger";

    } else {

        resultStatusBadge.textContent =
            "Safe Message";


        resultStatusBadge.className =
            "result-badge safe";


        predictionStatus.textContent =
            "Safe";


        predictionStatus.className =
            "metric-value safe";

    }


    // --------------------------------------------------------
    // SUMMARY
    // --------------------------------------------------------

    summaryText.textContent =
        data.summary ||
        (
            isCyberbullying
                ? "The message was classified as potentially harmful."
                : "The message was classified as non-harmful."
        );


    // --------------------------------------------------------
    // CONFIDENCE
    // --------------------------------------------------------

    const confidence =
        Number(
            data.confidence || 0
        );


    const safeConfidence =
        Math.min(
            Math.max(
                confidence,
                0
            ),
            1
        );


    const confidencePercentage =
        Math.round(
            safeConfidence * 100
        );


    confidenceValue.textContent =
        `${confidencePercentage}%`;


    confidenceFill.style.width =
        `${confidencePercentage}%`;


    // --------------------------------------------------------
    // SEVERITY
    // --------------------------------------------------------

    severityValue.textContent =
        formatLabel(
            data.severity
        );


    // --------------------------------------------------------
    // TARGET
    // --------------------------------------------------------

    targetValue.textContent =
        formatLabel(
            data.target_type
        );


    // --------------------------------------------------------
    // CATEGORIES
    // --------------------------------------------------------

    renderCategories(
        data.categories
    );


    // --------------------------------------------------------
    // EXPLANATION
    // --------------------------------------------------------

    reasonText.textContent =
        data.reason ||
        "No explanation available.";


    // --------------------------------------------------------
    // SEVERITY REASON
    // --------------------------------------------------------

    severityReason.textContent =
        data.severity_reason ||
        "No severity explanation available.";


    // --------------------------------------------------------
    // TARGET REASON
    // --------------------------------------------------------

    targetReason.textContent =
        data.target_reason ||
        "No target explanation available.";


    // --------------------------------------------------------
    // POLITE SUGGESTION
    // --------------------------------------------------------

    suggestionText.textContent =
        data.polite_suggestion ||
        "No suggestion available.";

}


// ============================================================
// RENDER CATEGORIES
// ============================================================

function renderCategories(categories) {

    categoryContainer.innerHTML =
        "";


    if (
        !Array.isArray(categories) ||
        categories.length === 0
    ) {

        const empty =
            document.createElement(
                "span"
            );


        empty.className =
            "empty-category";


        empty.textContent =
            "No categories detected";


        categoryContainer.appendChild(
            empty
        );


        return;

    }


    categories.forEach(
        (category) => {

            const badge =
                document.createElement(
                    "span"
                );


            badge.className =
                "category-badge";


            badge.textContent =
                formatLabel(
                    category
                );


            categoryContainer.appendChild(
                badge
            );

        }
    );

}


// ============================================================
// UPDATE STATISTICS
// ============================================================

function updateStatistics(data) {

    totalAnalysis++;


    const isCyberbullying =
        Boolean(
            data.cyberbullying
        );


    const confidence =
        Number(
            data.confidence || 0
        );


    totalConfidence +=
        confidence;


    if (isCyberbullying) {

        cyberbullyingCount++;

    } else {

        safeCount++;

    }


    // --------------------------------------------------------
    // MAIN STATISTICS
    // --------------------------------------------------------

    document.getElementById(
        "totalAnalysis"
    ).textContent =
        totalAnalysis;


    document.getElementById(
        "cyberbullyingCount"
    ).textContent =
        cyberbullyingCount;


    document.getElementById(
        "safeCount"
    ).textContent =
        safeCount;


    const averageConfidence =
        totalAnalysis > 0
            ? totalConfidence /
              totalAnalysis
            : 0;


    document.getElementById(
        "averageConfidence"
    ).textContent =
        `${Math.round(
            averageConfidence * 100
        )}%`;


    // --------------------------------------------------------
    // ANALYTICS
    // --------------------------------------------------------

    document.getElementById(
        "analyticsTotal"
    ).textContent =
        totalAnalysis;


    document.getElementById(
        "analyticsDetected"
    ).textContent =
        cyberbullyingCount;


    document.getElementById(
        "analyticsSafe"
    ).textContent =
        safeCount;


    const detectionRate =
        totalAnalysis > 0
            ? (
                cyberbullyingCount /
                totalAnalysis
            ) * 100
            : 0;


    document.getElementById(
        "detectionRate"
    ).textContent =
        `${Math.round(
            detectionRate
        )}%`;

}


// ============================================================
// HISTORY
// ============================================================

function addToHistory(
    text,
    data
) {

    const historyItem = {

        text: text,

        cyberbullying:
            Boolean(
                data.cyberbullying
            ),

        confidence:
            Number(
                data.confidence || 0
            ),

        severity:
            data.severity ||
            "none",

        timestamp:
            new Date()

    };


    detectionHistory.unshift(
        historyItem
    );


    // Keep latest 10

    if (
        detectionHistory.length > 10
    ) {

        detectionHistory =
            detectionHistory.slice(
                0,
                10
            );

    }


    renderHistory();

}


// ============================================================
// RENDER HISTORY
// ============================================================

function renderHistory() {

    historyContainer.innerHTML =
        "";


    if (
        detectionHistory.length === 0
    ) {

        historyContainer.innerHTML = `
            <div class="empty-history">
                No messages analyzed yet.
            </div>
        `;

        return;

    }


    detectionHistory.forEach(
        (item) => {

            const historyElement =
                document.createElement(
                    "div"
                );


            historyElement.className =
                "history-item";


            const statusClass =
                item.cyberbullying
                    ? "danger"
                    : "safe";


            const statusText =
                item.cyberbullying
                    ? "Cyberbullying"
                    : "Safe";


            const confidence =
                Math.round(
                    item.confidence * 100
                );


            const time =
                formatTime(
                    item.timestamp
                );


            historyElement.innerHTML = `

                <div
                    class="history-status ${statusClass}"
                ></div>

                <div class="history-content">

                    <div class="history-text">
                        ${escapeHtml(
                            item.text
                        )}
                    </div>

                    <div class="history-meta">

                        <span>
                            ${statusText}
                        </span>

                        <span>
                            Severity:
                            ${formatLabel(
                                item.severity
                            )}
                        </span>

                        <span>
                            ${time}
                        </span>

                    </div>

                </div>

                <div class="history-confidence">
                    ${confidence}%
                </div>

            `;


            historyContainer.appendChild(
                historyElement
            );

        }
    );

}


// ============================================================
// CLEAR HISTORY
// ============================================================

clearHistoryButton.addEventListener(
    "click",
    () => {

        detectionHistory = [];

        lastPrediction = null;

        renderHistory();


        // Clear stored session

        saveSessionData();


        // Reset result panel

        resultDetails.classList.add(
            "hidden"
        );


        resultMain.classList.remove(
            "hidden"
        );


        resultStatusBadge.textContent =
            "Waiting";


        resultStatusBadge.className =
            "result-badge neutral";


        // Clear input

        messageInput.value =
            "";


        characterCount.textContent =
            "0 / 1000";

    }
);


// ============================================================
// LOADING STATE
// ============================================================

function setLoading(isLoading) {

    if (isLoading) {

        analyzeButton.disabled =
            true;


        analyzeButton.innerHTML = `
            <span class="spinner"></span>
            Analyzing...
        `;


        loadingState.classList.remove(
            "hidden"
        );

    } else {

        analyzeButton.disabled =
            false;


        analyzeButton.innerHTML = `
            <span class="button-icon">✦</span>
            Analyze Message
            <span class="button-arrow">→</span>
        `;


        loadingState.classList.add(
            "hidden"
        );

    }

}


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(message) {

    errorMessage.textContent =
        message;


    errorMessage.classList.remove(
        "hidden"
    );

}


function hideError() {

    errorMessage.classList.add(
        "hidden"
    );


    errorMessage.textContent =
        "";

}


// ============================================================
// FORMAT LABEL
// ============================================================

function formatLabel(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return "None";

    }


    return String(value)
        .replaceAll(
            "_",
            " "
        )
        .replace(
            /\b\w/g,
            (letter) =>
                letter.toUpperCase()
        );

}


// ============================================================
// FORMAT TIME
// ============================================================

function formatTime(date) {

    return date.toLocaleTimeString(
        [],
        {
            hour: "2-digit",
            minute: "2-digit"
        }
    );

}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(value) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        value;


    return div.innerHTML;

}


// ============================================================
// API HEALTH CHECK
// ============================================================

async function checkApiHealth() {

    try {

        const response =
            await fetch(
                "/health"
            );


        if (!response.ok) {

            throw new Error(
                "API unavailable"
            );

        }


        const data =
            await response.json();


        console.log(
            "Cyberbullying Detection API connected."
        );


        console.log(
            "Model:",
            data.model
        );


        console.log(
            "Pipeline:",
            data.pipeline
        );


        console.log(
            "Feature Dimension:",
            data.feature_dimension
        );


        if (apiStatus) {

            apiStatus.innerHTML = `
                <span class="status-dot"></span>
                API Connected
            `;

        }


    } catch (error) {

        console.warn(
            "API health check failed:",
            error
        );


        if (apiStatus) {

            apiStatus.innerHTML = `
                <span class="status-dot"></span>
                API Offline
            `;

        }

    }

}


// ============================================================
// INITIALIZE
// ============================================================

loadSessionData();

restoreSessionUI();

checkApiHealth();