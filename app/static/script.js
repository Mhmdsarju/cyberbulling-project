/* ============================================================
   CYBERBULLYING DETECTION SYSTEM
   FRONTEND SCRIPT
============================================================ */


/* ============================================================
   GLOBAL STATE
============================================================ */

let totalAnalysis = 0;
let cyberbullyingCount = 0;
let safeCount = 0;
let totalConfidence = 0;

let detectionHistory = [];

let lastPrediction = null;

const SESSION_STORAGE_KEY =
    "cyberbullying_detection_session";


/* ============================================================
   DOM HELPERS
============================================================ */

function getElement(id) {

    return document.getElementById(id);

}


/* ============================================================
   DOM ELEMENTS
============================================================ */

const messageInput =
    getElement("messageInput");

const analyzeButton =
    getElement("analyzeButton");

const clearButton =
    getElement("clearButton");

const characterCount =
    getElement("characterCount");

const loadingState =
    getElement("loadingState");

const errorMessage =
    getElement("errorMessage");

const apiStatus =
    getElement("apiStatus");

const sidebarApiDot =
    getElement("sidebarApiDot");

const resultStatusBadge =
    getElement("resultStatusBadge");

const resultMain =
    getElement("resultMain");

const resultDetails =
    getElement("resultDetails");

const summaryText =
    getElement("summaryText");

const predictionStatus =
    getElement("predictionStatus");

const confidenceValue =
    getElement("confidenceValue");

const confidenceFill =
    getElement("confidenceFill");

const severityValue =
    getElement("severityValue");

const targetValue =
    getElement("targetValue");

const intentValue =
    getElement("intentValue");

const contentCategoryValue =
    getElement("contentCategoryValue");

const categoryContainer =
    getElement("categoryContainer");

const reasonText =
    getElement("reasonText");

const severityReason =
    getElement("severityReason");

const targetReason =
    getElement("targetReason");

const suggestionText =
    getElement("suggestionText");

const historyContainer =
    getElement("historyContainer");

const recentActivity =
    getElement("recentActivity");

const clearHistoryButton =
    getElement("clearHistoryButton");


/* ============================================================
   SESSION STORAGE
============================================================ */

function saveSessionData() {

    const sessionData = {

        totalAnalysis,

        cyberbullyingCount,

        safeCount,

        totalConfidence,

        detectionHistory,

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


/* ============================================================
   LOAD SESSION DATA
============================================================ */

function loadSessionData() {

    try {

        const savedData =
            sessionStorage.getItem(
                SESSION_STORAGE_KEY
            );


        if (!savedData) {

            return;

        }


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


/* ============================================================
   UPDATE ALL STATISTICS
============================================================ */

function updateStatisticsUI() {

    const averageConfidence =
        totalAnalysis > 0
            ? totalConfidence / totalAnalysis
            : 0;


    const detectionRate =
        totalAnalysis > 0
            ? (
                cyberbullyingCount /
                totalAnalysis
            ) * 100
            : 0;


    /* Dashboard */

    setText(
        "totalAnalysis",
        totalAnalysis
    );


    setText(
        "cyberbullyingCount",
        cyberbullyingCount
    );


    setText(
        "safeCount",
        safeCount
    );


    setText(
        "averageConfidence",
        `${Math.round(
            averageConfidence * 100
        )}%`
    );


    /* Analytics */

    setText(
        "analyticsTotal",
        totalAnalysis
    );


    setText(
        "analyticsDetected",
        cyberbullyingCount
    );


    setText(
        "analyticsSafe",
        safeCount
    );


    setText(
        "detectionRate",
        `${Math.round(
            detectionRate
        )}%`
    );


    /* Other possible dashboard IDs */

    setText(
        "totalDetections",
        totalAnalysis
    );


    setText(
        "detectedCount",
        cyberbullyingCount
    );


    setText(
        "safeMessages",
        safeCount
    );


    setText(
        "confidenceScore",
        `${Math.round(
            averageConfidence * 100
        )}%`
    );

}


/* ============================================================
   SET TEXT SAFELY
============================================================ */

function setText(id, value) {

    const element =
        getElement(id);


    if (element) {

        element.textContent =
            value;

    }

}


/* ============================================================
   TEXT NORMALIZATION
============================================================ */

function normalizeInputText(value) {

    return String(value || "")
        .normalize("NFKC")
        .replace(/\s+/g, " ")
        .trim();

}


/* ============================================================
   RESTORE SESSION UI
============================================================ */

function restoreSessionUI() {

    updateStatisticsUI();

    renderHistory();

    renderRecentActivity();


    if (
        lastPrediction &&
        messageInput
    ) {

        if (lastPrediction.text) {

            messageInput.value =
                lastPrediction.text;


            updateCharacterCount();

        }


        if (lastPrediction.data) {

            displayPrediction(
                lastPrediction.data
            );

        }

    }

}


/* ============================================================
   CHARACTER COUNTER
============================================================ */

function updateCharacterCount() {

    if (!messageInput) {

        return;

    }


    const length =
        messageInput.value.length;


    if (characterCount) {

        characterCount.textContent =
            `${length} / 1000`;

    }

}


/* ============================================================
   INPUT EVENTS
============================================================ */

if (messageInput) {

    messageInput.addEventListener(
        "input",
        updateCharacterCount
    );


    messageInput.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                (
                    event.ctrlKey ||
                    event.metaKey
                )
            ) {

                event.preventDefault();

                analyzeMessage();

            }

        }
    );

}


/* ============================================================
   CLEAR INPUT
============================================================ */

if (clearButton) {

    clearButton.addEventListener(
        "click",
        () => {

            if (messageInput) {

                messageInput.value = "";

                updateCharacterCount();

                messageInput.focus();

            }


            hideError();

        }
    );

}


/* ============================================================
   EXAMPLE BUTTONS
============================================================ */

const exampleButtons =
    document.querySelectorAll(
        ".example-button"
    );


exampleButtons.forEach(
    (button) => {

        button.addEventListener(
            "click",
            () => {

                if (!messageInput) {

                    return;

                }


                const text =
                    button.dataset.text ||
                    button.textContent.trim();


                messageInput.value =
                    normalizeInputText(text);


                updateCharacterCount();

                hideError();

                messageInput.focus();

            }
        );

    }
);


/* ============================================================
   ANALYZE BUTTON
============================================================ */

if (analyzeButton) {

    analyzeButton.addEventListener(
        "click",
        analyzeMessage
    );

}


/* ============================================================
   ANALYZE MESSAGE
============================================================ */

async function analyzeMessage() {

    if (!messageInput) {

        return;

    }


    const text =
        normalizeInputText(
            messageInput.value
        );


    hideError();


    /* --------------------------------------------------------
       VALIDATION
    -------------------------------------------------------- */

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


    messageInput.value =
        text;

    updateCharacterCount();


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

                    body:
                        JSON.stringify({

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


        /* ----------------------------------------------------
           SAVE LATEST PREDICTION
        ---------------------------------------------------- */

        lastPrediction = {

            text,

            data

        };


        /* ----------------------------------------------------
           DISPLAY RESULT
        ---------------------------------------------------- */

        displayPrediction(
            data
        );


        /* ----------------------------------------------------
           UPDATE STATISTICS
        ---------------------------------------------------- */

        updateStatistics(
            data
        );


        /* ----------------------------------------------------
           ADD HISTORY
        ---------------------------------------------------- */

        addToHistory(
            text,
            data
        );


        /* ----------------------------------------------------
           SAVE SESSION
        ---------------------------------------------------- */

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

        setLoading(false);

    }

}


/* ============================================================
   DISPLAY PREDICTION
============================================================ */

function displayPrediction(data) {

    const isCyberbullying =
        Boolean(
            data.cyberbullying
        );


    /* --------------------------------------------------------
       RESULT VISIBILITY
    -------------------------------------------------------- */

    if (resultMain) {

        resultMain.classList.add(
            "hidden"
        );

    }


    if (resultDetails) {

        resultDetails.classList.remove(
            "hidden"
        );

    }


    /* --------------------------------------------------------
       STATUS BADGE
    -------------------------------------------------------- */

    if (resultStatusBadge) {

        if (isCyberbullying) {

            resultStatusBadge.textContent =
                "Cyberbullying Detected";


            resultStatusBadge.className =
                "result-badge danger";

        } else {

            resultStatusBadge.textContent =
                "Safe Message";


            resultStatusBadge.className =
                "result-badge safe";

        }

    }


    /* --------------------------------------------------------
       PREDICTION STATUS
    -------------------------------------------------------- */

    if (predictionStatus) {

        predictionStatus.textContent =
            isCyberbullying
                ? "Detected"
                : "Safe";


        predictionStatus.className =
            isCyberbullying
                ? "metric-value danger"
                : "metric-value safe";

    }


    /* --------------------------------------------------------
       SUMMARY
    -------------------------------------------------------- */

    if (summaryText) {

        summaryText.textContent =
            data.summary ||
            (
                isCyberbullying
                    ? "The message was classified as potentially harmful."
                    : "The message was classified as non-harmful."
            );

    }


    /* --------------------------------------------------------
       CONFIDENCE
    -------------------------------------------------------- */

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


    if (confidenceValue) {

        confidenceValue.textContent =
            `${confidencePercentage}%`;

    }


    if (confidenceFill) {

        confidenceFill.style.width =
            `${confidencePercentage}%`;

    }


    /* --------------------------------------------------------
       SEVERITY
    -------------------------------------------------------- */

    if (severityValue) {

        severityValue.textContent =
            formatLabel(
                data.severity
            );

    }


    /* --------------------------------------------------------
       TARGET
    -------------------------------------------------------- */

    if (targetValue) {

        targetValue.textContent =
            formatLabel(
                data.target_type
            );

    }


    /* --------------------------------------------------------
       INTENT
    -------------------------------------------------------- */

    if (intentValue) {

        intentValue.textContent =
            formatLabel(
                data.intent
            );

    }


    /* --------------------------------------------------------
       CONTENT CATEGORY
    -------------------------------------------------------- */

    if (contentCategoryValue) {

        contentCategoryValue.textContent =
            formatLabel(
                data.content_category
            );

    }


    /* --------------------------------------------------------
       OFFENSE CATEGORIES
    -------------------------------------------------------- */

    renderCategories(
        data.categories
    );


    /* --------------------------------------------------------
       EXPLANATION
    -------------------------------------------------------- */

    if (reasonText) {

        reasonText.textContent =
            data.reason ||
            "No explanation available.";

    }


    /* --------------------------------------------------------
       SEVERITY REASON
    -------------------------------------------------------- */

    if (severityReason) {

        severityReason.textContent =
            data.severity_reason ||
            "No severity explanation available.";

    }


    /* --------------------------------------------------------
       TARGET REASON
    -------------------------------------------------------- */

    if (targetReason) {

        targetReason.textContent =
            data.target_reason ||
            "No target explanation available.";

    }


    /* --------------------------------------------------------
       POLITE SUGGESTION
    -------------------------------------------------------- */

    if (suggestionText) {

        suggestionText.textContent =
            data.polite_suggestion ||
            "No suggestion available.";

    }

}


/* ============================================================
   RENDER OFFENSE CATEGORIES
============================================================ */

function renderCategories(categories) {

    if (!categoryContainer) {

        return;

    }


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
                "category-tag";


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


/* ============================================================
   UPDATE STATISTICS
============================================================ */

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


    updateStatisticsUI();

}


/* ============================================================
   ADD HISTORY
============================================================ */

function addToHistory(
    text,
    data
) {

    const historyItem = {

        text,

        cyberbullying:
            Boolean(
                data.cyberbullying
            ),

        confidence:
            Number(
                data.confidence || 0
            ),

        categories:
            Array.isArray(
                data.categories
            )
                ? data.categories
                : [],

        intent:
            data.intent ||
            "none",

        content_category:
            data.content_category ||
            "none",

        severity:
            data.severity ||
            "none",

        target_type:
            data.target_type ||
            "none",

        timestamp:
            new Date()

    };


    detectionHistory.unshift(
        historyItem
    );


    /* Keep latest 10 */

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

    renderRecentActivity();

}


/* ============================================================
   RENDER ANALYSIS PAGE HISTORY
============================================================ */

function renderHistory() {

    if (!historyContainer) {

        return;

    }


    historyContainer.innerHTML =
        "";


    if (
        detectionHistory.length === 0
    ) {

        historyContainer.innerHTML = `
            <div class="empty-state">

                <div>
                    ⌁
                </div>

                <p>
                    No messages analyzed yet.
                </p>

                <small>
                    Your recent detections will appear here.
                </small>

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
                    Number(
                        item.confidence || 0
                    ) * 100
                );


            const categoryText =
                Array.isArray(
                    item.categories
                ) &&
                item.categories.length > 0
                    ? item.categories
                        .map(formatLabel)
                        .join(", ")
                    : "None";


            const time =
                formatTime(
                    item.timestamp
                );


            historyElement.innerHTML = `

                <div class="history-content">

                    <div class="history-text">
                        ${escapeHtml(
                            item.text
                        )}
                    </div>

                    <div class="history-meta">

                        <span
                            class="history-status ${statusClass}"
                        >
                            ${statusText}
                        </span>

                        <span>
                            Severity:
                            ${formatLabel(
                                item.severity
                            )}
                        </span>

                        <span>
                            Category:
                            ${escapeHtml(
                                categoryText
                            )}
                        </span>

                        <span>
                            Intent:
                            ${formatLabel(
                                item.intent
                            )}
                        </span>

                        <span>
                            Content:
                            ${formatLabel(
                                item.content_category
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


/* ============================================================
   RENDER DASHBOARD RECENT ACTIVITY
============================================================ */

function renderRecentActivity() {

    if (!recentActivity) {

        return;

    }


    recentActivity.innerHTML =
        "";


    if (
        !detectionHistory ||
        detectionHistory.length === 0
    ) {

        recentActivity.innerHTML = `
            <div class="empty-state">

                <div>
                    ◷
                </div>

                <p>
                    No analysis performed yet.
                </p>

                <small>
                    Your recent predictions will appear here.
                </small>

            </div>
        `;

        return;

    }


    /* Show latest 5 */

    const recentItems =
        detectionHistory.slice(
            0,
            5
        );


    recentItems.forEach(
        (item) => {

            const activity =
                document.createElement(
                    "div"
                );


            activity.className =
                "recent-activity-item";


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
                    Number(
                        item.confidence || 0
                    ) * 100
                );


            const categoryText =
                Array.isArray(
                    item.categories
                ) &&
                item.categories.length > 0
                    ? item.categories
                        .map(formatLabel)
                        .join(", ")
                    : "No category";


            activity.innerHTML = `

                <div class="recent-activity-main">

                    <div class="recent-activity-text">

                        ${escapeHtml(
                            item.text
                        )}

                    </div>

                    <div class="recent-activity-meta">

                        <span
                            class="history-status ${statusClass}"
                        >
                            ${statusText}
                        </span>

                        <span>
                            ${escapeHtml(
                                categoryText
                            )}
                        </span>

                        <span>
                            Intent:
                            ${formatLabel(
                                item.intent
                            )}
                        </span>

                        <span>
                            Content:
                            ${formatLabel(
                                item.content_category
                            )}
                        </span>

                        <span>
                            ${formatTime(
                                item.timestamp
                            )}
                        </span>

                    </div>

                </div>


                <div class="recent-activity-confidence">

                    ${confidence}%

                </div>

            `;


            recentActivity.appendChild(
                activity
            );

        }
    );

}


/* ============================================================
   CLEAR HISTORY
============================================================ */

if (clearHistoryButton) {

    clearHistoryButton.addEventListener(
        "click",
        () => {

            detectionHistory = [];

            lastPrediction = null;

            totalAnalysis = 0;

            cyberbullyingCount = 0;

            safeCount = 0;

            totalConfidence = 0;


            renderHistory();

            renderRecentActivity();

            updateStatisticsUI();


            if (messageInput) {

                messageInput.value = "";

                updateCharacterCount();

            }


            if (resultDetails) {

                resultDetails.classList.add(
                    "hidden"
                );

            }


            if (resultMain) {

                resultMain.classList.remove(
                    "hidden"
                );

            }


            if (resultStatusBadge) {

                resultStatusBadge.textContent =
                    "Waiting";


                resultStatusBadge.className =
                    "result-badge neutral";

            }


            saveSessionData();

        }
    );

}


/* ============================================================
   LOADING STATE
============================================================ */

function setLoading(isLoading) {

    if (!analyzeButton) {

        return;

    }


    if (isLoading) {

        analyzeButton.disabled =
            true;


        analyzeButton.innerHTML = `
            <span class="spinner"></span>
            Analyzing...
        `;


        if (loadingState) {

            loadingState.classList.remove(
                "hidden"
            );

        }

    } else {

        analyzeButton.disabled =
            false;


        analyzeButton.innerHTML = `
            <span class="button-icon">✦</span>
            Analyze Message
            <span class="button-arrow">→</span>
        `;


        if (loadingState) {

            loadingState.classList.add(
                "hidden"
            );

        }

    }

}


/* ============================================================
   ERROR HANDLING
============================================================ */

function showError(message) {

    if (!errorMessage) {

        return;

    }


    errorMessage.textContent =
        message;


    errorMessage.classList.remove(
        "hidden"
    );

}


function hideError() {

    if (!errorMessage) {

        return;

    }


    errorMessage.classList.add(
        "hidden"
    );


    errorMessage.textContent =
        "";

}


/* ============================================================
   FORMAT LABEL
============================================================ */

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


/* ============================================================
   FORMAT TIME
============================================================ */

function formatTime(date) {

    try {

        const parsedDate =
            date instanceof Date
                ? date
                : new Date(date);


        if (
            Number.isNaN(
                parsedDate.getTime()
            )
        ) {

            return "--:--";

        }


        return parsedDate.toLocaleTimeString(
            [],
            {

                hour: "2-digit",

                minute: "2-digit"

            }
        );

    } catch {

        return "--:--";

    }

}


/* ============================================================
   HTML ESCAPE
============================================================ */

function escapeHtml(value) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        String(value);


    return div.innerHTML;

}


/* ============================================================
   API HEALTH CHECK
============================================================ */

async function checkApiHealth() {

    if (!apiStatus) {

        return;

    }


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
            "Prediction API connected."
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
            "Prediction Components:",
            data.prediction_components
        );


        console.log(
            "Feature Dimension:",
            data.feature_dimension
        );


        apiStatus.innerHTML = `
            <span class="online-dot"></span>
            API Connected
        `;


        if (sidebarApiDot) {

            sidebarApiDot.classList.add(
                "online-dot"
            );

        }


    } catch (error) {

        console.warn(
            "API health check failed:",
            error
        );


        apiStatus.innerHTML = `
            <span
                class="online-dot"
                style="background:#ff5d73"
            ></span>
            API Offline
        `;


        if (sidebarApiDot) {

            sidebarApiDot.style.background =
                "#ff5d73";

        }

    }

}


/* ============================================================
   PAGE INITIALIZATION
============================================================ */

function initializePage() {

    loadSessionData();

    restoreSessionUI();

    checkApiHealth();

    updateCharacterCount();

}


/* ============================================================
   START
============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    initializePage
);