import re


# ============================================================
# CONFIGURATION
# ============================================================

VALID_CATEGORIES = {
    "insult",
    "threat",
    "violence",
    "hate",
    "sexual_harassment",
    "sexual_abuse",
    "profanity",
}

VALID_SEVERITIES = {
    "none",
    "low",
    "medium",
    "high",
}

VALID_TARGET_TYPES = {
    "none",
    "individual",
    "group",
    "other",
}


# ============================================================
# CATEGORY EXPLANATIONS
# ============================================================

CATEGORY_REASONS = {

    "insult": (
        "The text contains insulting or degrading language "
        "directed at a person or target."
    ),

    "threat": (
        "The text contains language that indicates a threat "
        "or intention to cause harm."
    ),

    "violence": (
        "The text contains language associated with physical "
        "violence or harm."
    ),

    "hate": (
        "The text contains hateful or discriminatory language "
        "towards a group or target."
    ),

    "sexual_harassment": (
        "The text contains unwanted or inappropriate sexual "
        "language directed towards a target."
    ),

    "sexual_abuse": (
        "The text contains abusive sexual language or references "
        "associated with sexual abuse."
    ),

    "profanity": (
        "The text contains offensive or profane language."
    ),
}


# ============================================================
# SEVERITY EXPLANATIONS
# ============================================================

SEVERITY_REASONS = {

    "none": (
        "No significant cyberbullying severity was detected."
    ),

    "low": (
        "The detected content indicates a lower level of "
        "offensive or harmful behavior."
    ),

    "medium": (
        "The detected content indicates a moderate level of "
        "harmful or offensive behavior."
    ),

    "high": (
        "The detected content indicates a high level of "
        "harmful, threatening, or abusive behavior."
    ),
}


# ============================================================
# TARGET EXPLANATIONS
# ============================================================

TARGET_REASONS = {

    "none": (
        "No specific target was identified."
    ),

    "individual": (
        "The content appears to be directed towards an individual."
    ),

    "group": (
        "The content appears to be directed towards a group."
    ),

    "other": (
        "The content appears to target another type of entity "
        "or target."
    ),
}


# ============================================================
# CATEGORY DISPLAY NAMES
# ============================================================

CATEGORY_DISPLAY_NAMES = {

    "insult": "Insult",

    "threat": "Threat",

    "violence": "Violence",

    "hate": "Hate",

    "sexual_harassment": "Sexual Harassment",

    "sexual_abuse": "Sexual Abuse",

    "profanity": "Profanity",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_categories(categories):

    if categories is None:
        return []

    if isinstance(categories, str):
        categories = [categories]

    normalized = []

    for category in categories:

        category = str(
            category
        ).strip().lower()

        if (
            category in VALID_CATEGORIES
            and category not in normalized
        ):

            normalized.append(
                category
            )

    return normalized


# ============================================================
# SEVERITY NORMALIZATION
# ============================================================

def normalize_severity(severity):

    if severity is None:
        return "none"

    severity = str(
        severity
    ).strip().lower()

    if severity in VALID_SEVERITIES:
        return severity

    return "none"


# ============================================================
# TARGET TYPE NORMALIZATION
# ============================================================

def normalize_target_type(target_type):

    if target_type is None:
        return "none"

    target_type = str(
        target_type
    ).strip().lower()

    if target_type in VALID_TARGET_TYPES:
        return target_type

    return "none"


# ============================================================
# CONFIDENCE NORMALIZATION
# ============================================================

def normalize_confidence(confidence):

    try:

        confidence = float(
            confidence
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            confidence
        )
    )


# ============================================================
# CYBERBULLYING NORMALIZATION
# ============================================================

def normalize_cyberbullying(value):

    if isinstance(value, bool):
        return value

    try:

        return bool(
            int(value)
        )

    except (
        TypeError,
        ValueError
    ):

        return False


# ============================================================
# CATEGORY REASON BUILDER
# ============================================================

def build_category_reason(categories):

    categories = normalize_categories(
        categories
    )

    if not categories:

        return (
            "No specific offensive category was detected."
        )

    reasons = []

    for category in categories:

        reason = CATEGORY_REASONS.get(
            category,
            "The text contains potentially harmful content."
        )

        reasons.append(
            reason
        )

    return " ".join(
        reasons
    )


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================

def generate_explanation(
    text,
    cyberbullying,
    categories,
    severity,
    target_type
):

    text = normalize_text(
        text
    )

    categories = normalize_categories(
        categories
    )

    severity = normalize_severity(
        severity
    )

    target_type = normalize_target_type(
        target_type
    )

    cyberbullying = normalize_cyberbullying(
        cyberbullying
    )

    # --------------------------------------------------------
    # NON-CYBERBULLYING
    # --------------------------------------------------------

    if not cyberbullying:

        return {

            "text": text,

            "cyberbullying": False,

            "summary": (
                "The text was classified as "
                "non-cyberbullying."
            ),

            "reason": (
                "No significant cyberbullying indicators "
                "were detected in the text."
            ),

            "category_reason": (
                "No specific offensive category was detected."
            ),

            "severity_reason": (
                SEVERITY_REASONS["none"]
            ),

            "target_reason": (
                TARGET_REASONS["none"]
            ),
        }

    # --------------------------------------------------------
    # CYBERBULLYING
    # --------------------------------------------------------

    category_reason = build_category_reason(
        categories
    )

    severity_reason = SEVERITY_REASONS.get(
        severity,
        SEVERITY_REASONS["none"]
    )

    target_reason = TARGET_REASONS.get(
        target_type,
        TARGET_REASONS["none"]
    )

    category_names = [

        CATEGORY_DISPLAY_NAMES.get(
            category,
            category
        )

        for category in categories
    ]

    if category_names:

        category_text = ", ".join(
            category_names
        )

        summary = (
            f"Cyberbullying detected. "
            f"Detected category: {category_text}. "
            f"Severity: {severity}. "
            f"Target: {target_type}."
        )

    else:

        summary = (
            f"Cyberbullying detected. "
            f"Severity: {severity}. "
            f"Target: {target_type}."
        )

    return {

        "text": text,

        "cyberbullying": True,

        "summary": summary,

        "reason": category_reason,

        "category_reason": category_reason,

        "severity_reason": severity_reason,

        "target_reason": target_reason,
    }


# ============================================================
# POLITE SUGGESTION
# ============================================================

def generate_polite_suggestion(
    cyberbullying,
    categories
):

    categories = normalize_categories(
        categories
    )

    cyberbullying = normalize_cyberbullying(
        cyberbullying
    )

    if not cyberbullying:

        return (
            "The message appears acceptable. "
            "Continue communicating respectfully."
        )

    if "threat" in categories:

        return (
            "Please express your concern without threats "
            "or language that could make another person "
            "feel unsafe."
        )

    if "hate" in categories:

        return (
            "Please avoid discriminatory language and "
            "express your disagreement respectfully."
        )

    if (
        "sexual_harassment" in categories
        or
        "sexual_abuse" in categories
    ):

        return (
            "Please avoid inappropriate sexual language "
            "and communicate respectfully."
        )

    if "violence" in categories:

        return (
            "Please avoid violent language and "
            "express your disagreement peacefully."
        )

    if (
        "insult" in categories
        or
        "profanity" in categories
    ):

        return (
            "Please avoid insulting or offensive language "
            "and communicate respectfully."
        )

    return (
        "Please communicate respectfully and avoid "
        "harmful or offensive language."
    )


# ============================================================
# COMPLETE RESULT BUILDER
# ============================================================

def build_prediction_result(
    text,
    cyberbullying,
    categories,
    severity,
    target_type,
    confidence
):

    text = normalize_text(
        text
    )

    categories = normalize_categories(
        categories
    )

    severity = normalize_severity(
        severity
    )

    target_type = normalize_target_type(
        target_type
    )

    confidence = normalize_confidence(
        confidence
    )

    cyberbullying = normalize_cyberbullying(
        cyberbullying
    )

    explanation = generate_explanation(

        text=text,

        cyberbullying=cyberbullying,

        categories=categories,

        severity=severity,

        target_type=target_type,
    )

    suggestion = generate_polite_suggestion(

        cyberbullying=cyberbullying,

        categories=categories
    )

    return {

        "text": text,

        "cyberbullying": cyberbullying,

        "categories": categories,

        "severity": severity,

        "target_type": target_type,

        "confidence": round(
            confidence,
            4
        ),

        "summary": explanation[
            "summary"
        ],

        "reason": explanation[
            "reason"
        ],

        "category_reason": explanation[
            "category_reason"
        ],

        "severity_reason": explanation[
            "severity_reason"
        ],

        "target_reason": explanation[
            "target_reason"
        ],

        "polite_suggestion": suggestion,
    }


# ============================================================
# TEST EXAMPLES
# ============================================================

def main():

    examples = [

        {
            "text": "nee romba loosu da",

            "cyberbullying": 1,

            "categories": [
                "insult"
            ],

            "severity": "medium",

            "target_type": "individual",

            "confidence": 0.91,
        },

        {
            "text": (
                "Ithu nalla useful information da"
            ),

            "cyberbullying": 0,

            "categories": [],

            "severity": "none",

            "target_type": "none",

            "confidence": 0.96,
        },

        {
            "text": (
                "unakku violence panniduven"
            ),

            "cyberbullying": 1,

            "categories": [
                "threat",
                "violence"
            ],

            "severity": "high",

            "target_type": "individual",

            "confidence": 0.94,
        },
    ]

    for index, example in enumerate(
        examples,
        start=1
    ):

        print(
            "\n" + "-" * 75
        )

        print(
            f"EXAMPLE {index}"
        )

        print(
            "-" * 75
        )

        result = build_prediction_result(

            text=example["text"],

            cyberbullying=example[
                "cyberbullying"
            ],

            categories=example[
                "categories"
            ],

            severity=example[
                "severity"
            ],

            target_type=example[
                "target_type"
            ],

            confidence=example[
                "confidence"
            ],
        )

        for key, value in result.items():

            print(
                f"{key:20s}: {value}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()