from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

INPUT_FILE = (
    RAW_DIR /
    "cyberbullying_25000_realworld_multilingual_v3.csv"
)


# ============================================================
# REPORT FILES
# ============================================================

VALIDATION_REPORT = (
    REPORTS_DIR /
    "dataset_validation.csv"
)

CATEGORY_REPORT = (
    REPORTS_DIR /
    "category_distribution.csv"
)

SEVERITY_REPORT = (
    REPORTS_DIR /
    "severity_distribution.csv"
)

TARGET_REPORT = (
    REPORTS_DIR /
    "target_type_distribution.csv"
)

LANGUAGE_REPORT = (
    REPORTS_DIR /
    "language_distribution.csv"
)

DATA_SOURCE_REPORT = (
    REPORTS_DIR /
    "data_source_distribution.csv"
)

INTENT_REPORT = (
    REPORTS_DIR /
    "intent_distribution.csv"
)

CONTEXT_REPORT = (
    REPORTS_DIR /
    "context_type_distribution.csv"
)

ENVIRONMENT_REPORT = (
    REPORTS_DIR /
    "environment_distribution.csv"
)

PLATFORM_REPORT = (
    REPORTS_DIR /
    "platform_distribution.csv"
)

MEDIA_TYPE_REPORT = (
    REPORTS_DIR /
    "media_type_distribution.csv"
)

CONTENT_CATEGORY_REPORT = (
    REPORTS_DIR /
    "content_category_distribution.csv"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_ROWS = 25000


EXPECTED_LANGUAGES = {
    "tanglish",
    "english"
}


EXPECTED_CONTEXT_TYPES = {
    "social_media",
    "non_social_realworld"
}


EXPECTED_ENVIRONMENTS = {
    "social_media",
    "gaming_chat",
    "personal_chat",
    "messaging",
    "workplace",
    "school_college",
    "review_feedback",
    "public_forum",
    "email",
    "community_chat"
}


EXPECTED_PLATFORMS = {
    "instagram",
    "youtube",
    "facebook",
    "socialmedia",
    "nonsocial"
}


EXPECTED_MEDIA_TYPES = {
    "post",
    "reel",
    "story",
    "photo",
    "video",
    "caption",
    "comment",
    "message"
}


EXPECTED_CYBERBULLYING = {
    0,
    1
}


EXPECTED_INTENTS = {
    "offensive",
    "defensive",
    "begging",
    "requesting",
    "apologizing",
    "supporting",
    "praising",
    "thanking",
    "greeting",
    "questioning",
    "informational",
    "casual",
    "neutral"
}


EXPECTED_CONTENT_CATEGORIES = {
    "none",
    "safe_communication",
    "negative_media_feedback",
    "bad_content",
    "low_quality",
    "negative_review",
    "spam",
    "misinformation",
    "privacy",
    "positive_media_feedback",
    "personal_attack",
    "profanity",
    "threat_violence",
    "hate_abuse",
    "sexual_abuse"
}


EXPECTED_SEVERITY = {
    "none",
    "low",
    "medium",
    "high"
}


EXPECTED_TARGET_TYPES = {
    "none",
    "individual",
    "group",
    "other"
}


EXPECTED_DATA_SOURCES = {
    "synthetic_curated_realworld_v3"
}


CATEGORY_COLUMNS = [
    "insult",
    "threat",
    "violence",
    "hate",
    "sexual_harassment",
    "sexual_abuse",
    "profanity"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_FILE}\n\n"
            "Place the 25K CSV inside data/raw/"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    return df


# ============================================================
# REQUIRED COLUMNS
# ============================================================

def check_columns(df):

    required_columns = [
        "id",
        "text",
        "language",
        "context_type",
        "environment",
        "platform",
        "media_type",
        "cyberbullying",
        "intent",
        "content_category",
        "target_type",
        "insult",
        "threat",
        "violence",
        "hate",
        "sexual_harassment",
        "sexual_abuse",
        "profanity",
        "severity",
        "reason",
        "data_source"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    return True


# ============================================================
# ROW COUNT
# ============================================================

def check_row_count(df):

    actual_rows = len(df)

    status = (
        "PASS"
        if actual_rows == EXPECTED_ROWS
        else "WARNING"
    )

    return {
        "check": "row_count",
        "expected": EXPECTED_ROWS,
        "actual": actual_rows,
        "status": status
    }


# ============================================================
# MISSING VALUES
# ============================================================

def check_missing_values(df):

    total_missing = int(
        df.isna().sum().sum()
    )

    status = (
        "PASS"
        if total_missing == 0
        else "WARNING"
    )

    return {
        "check": "missing_values",
        "expected": 0,
        "actual": total_missing,
        "status": status
    }


# ============================================================
# DUPLICATE VALUES
# ============================================================

def check_duplicates(df):

    duplicate_ids = int(
        df["id"]
        .duplicated()
        .sum()
    )

    duplicate_texts = int(
        df["text"]
        .duplicated()
        .sum()
    )

    return [
        {
            "check": "duplicate_ids",
            "expected": 0,
            "actual": duplicate_ids,
            "status": (
                "PASS"
                if duplicate_ids == 0
                else "WARNING"
            )
        },
        {
            "check": "duplicate_texts",
            "expected": 0,
            "actual": duplicate_texts,
            "status": (
                "PASS"
                if duplicate_texts == 0
                else "WARNING"
            )
        }
    ]


# ============================================================
# CYBERBULLYING VALIDATION
# ============================================================

def check_cyberbullying(df):

    values = set(
        pd.to_numeric(
            df["cyberbullying"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    invalid = values - EXPECTED_CYBERBULLYING

    return {
        "check": "cyberbullying_values",
        "expected": "0, 1",
        "actual": (
            ", ".join(
                str(int(value))
                for value in sorted(values)
            )
        ),
        "status": (
            "PASS"
            if len(invalid) == 0
            else "FAIL"
        )
    }


# ============================================================
# GENERIC STRING VALIDATION
# ============================================================

def validate_string_column(
    df,
    column,
    expected_values
):

    values = set(
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    invalid = values - expected_values

    return {
        "check": f"{column}_values",
        "expected": ", ".join(
            sorted(expected_values)
        ),
        "actual": ", ".join(
            sorted(values)
        ),
        "status": (
            "PASS"
            if len(invalid) == 0
            else "FAIL"
        )
    }


# ============================================================
# CATEGORY VALIDATION
# ============================================================

def check_categories(df):

    results = []

    for column in CATEGORY_COLUMNS:

        values = set(
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        invalid = values - {
            0,
            1
        }

        results.append({
            "check": f"{column}_values",
            "expected": "0, 1",
            "actual": ", ".join(
                str(int(value))
                for value in sorted(values)
            ),
            "status": (
                "PASS"
                if len(invalid) == 0
                else "FAIL"
            )
        })

    return results


# ============================================================
# TEXT VALIDATION
# ============================================================

def check_text(df):

    text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_count = int(
        (text == "").sum()
    )

    lengths = text.str.len()

    return {
        "empty_texts": empty_count,
        "minimum_length": int(
            lengths.min()
        ),
        "maximum_length": int(
            lengths.max()
        ),
        "mean_length": round(
            lengths.mean(),
            2
        ),
        "median_length": round(
            lengths.median(),
            2
        )
    }


# ============================================================
# DISTRIBUTION REPORT
# ============================================================

def save_distribution(
    df,
    column,
    output_file
):

    distribution = (
        df[column]
        .value_counts()
        .rename_axis(column)
        .reset_index(
            name="count"
        )
    )

    distribution["percentage"] = (
        distribution["count"]
        / len(df)
        * 100
    ).round(2)

    distribution.to_csv(
        output_file,
        index=False
    )


# ============================================================
# CATEGORY REPORT
# ============================================================

def save_category_report(df):

    rows = []

    for column in CATEGORY_COLUMNS:

        positive = int(
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        negative = (
            len(df)
            - positive
        )

        rows.append({
            "category": column,
            "positive": positive,
            "negative": negative,
            "positive_percentage": round(
                positive / len(df) * 100,
                2
            )
        })

    pd.DataFrame(
        rows
    ).to_csv(
        CATEGORY_REPORT,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("DATASET VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = load_dataset()

    print("\nDataset:")
    print(
        f"Rows    : {len(df)}"
    )
    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    check_columns(df)

    print(
        "\nRequired columns: PASS"
    )

    # --------------------------------------------------------
    # BASIC CHECKS
    # --------------------------------------------------------

    validation_results = []

    validation_results.append(
        check_row_count(df)
    )

    validation_results.append(
        check_missing_values(df)
    )

    validation_results.extend(
        check_duplicates(df)
    )

    validation_results.append(
        check_cyberbullying(df)
    )

    # --------------------------------------------------------
    # STRING LABEL CHECKS
    # --------------------------------------------------------

    validation_results.append(
        validate_string_column(
            df,
            "language",
            EXPECTED_LANGUAGES
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "context_type",
            EXPECTED_CONTEXT_TYPES
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "environment",
            EXPECTED_ENVIRONMENTS
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "platform",
            EXPECTED_PLATFORMS
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "media_type",
            EXPECTED_MEDIA_TYPES
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "intent",
            EXPECTED_INTENTS
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "content_category",
            EXPECTED_CONTENT_CATEGORIES
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "severity",
            EXPECTED_SEVERITY
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "target_type",
            EXPECTED_TARGET_TYPES
        )
    )

    validation_results.append(
        validate_string_column(
            df,
            "data_source",
            EXPECTED_DATA_SOURCES
        )
    )

    # --------------------------------------------------------
    # OFFENSE CATEGORY CHECK
    # --------------------------------------------------------

    validation_results.extend(
        check_categories(df)
    )

    # --------------------------------------------------------
    # TEXT CHECK
    # --------------------------------------------------------

    text_stats = check_text(df)

    validation_results.append({
        "check": "empty_texts",
        "expected": 0,
        "actual": text_stats["empty_texts"],
        "status": (
            "PASS"
            if text_stats["empty_texts"] == 0
            else "WARNING"
        )
    })

    # --------------------------------------------------------
    # SAVE DISTRIBUTIONS
    # --------------------------------------------------------

    save_category_report(df)

    save_distribution(
        df,
        "severity",
        SEVERITY_REPORT
    )

    save_distribution(
        df,
        "target_type",
        TARGET_REPORT
    )

    save_distribution(
        df,
        "language",
        LANGUAGE_REPORT
    )

    save_distribution(
        df,
        "data_source",
        DATA_SOURCE_REPORT
    )

    save_distribution(
        df,
        "intent",
        INTENT_REPORT
    )

    save_distribution(
        df,
        "context_type",
        CONTEXT_REPORT
    )

    save_distribution(
        df,
        "environment",
        ENVIRONMENT_REPORT
    )

    save_distribution(
        df,
        "platform",
        PLATFORM_REPORT
    )

    save_distribution(
        df,
        "media_type",
        MEDIA_TYPE_REPORT
    )

    save_distribution(
        df,
        "content_category",
        CONTENT_CATEGORY_REPORT
    )

    # --------------------------------------------------------
    # SAVE VALIDATION REPORT
    # --------------------------------------------------------

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df.to_csv(
        VALIDATION_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print("\nValidation Results:")

    print(
        validation_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TEXT STATISTICS
    # --------------------------------------------------------

    print("\nText Statistics:")

    print(
        f"  Empty   : {text_stats['empty_texts']}"
    )

    print(
        f"  Minimum : {text_stats['minimum_length']}"
    )

    print(
        f"  Maximum : {text_stats['maximum_length']}"
    )

    print(
        f"  Mean    : {text_stats['mean_length']}"
    )

    print(
        f"  Median  : {text_stats['median_length']}"
    )

    # --------------------------------------------------------
    # DATASET DISTRIBUTION
    # --------------------------------------------------------

    print("\nDataset Distribution:")

    print("\nCyberbullying:")

    print(
        df["cyberbullying"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nIntent:")

    print(
        df["intent"]
        .value_counts()
        .to_string()
    )

    print("\nContent Category:")

    print(
        df["content_category"]
        .value_counts()
        .to_string()
    )

    print("\nContext Type:")

    print(
        df["context_type"]
        .value_counts()
        .to_string()
    )

    print("\nEnvironment:")

    print(
        df["environment"]
        .value_counts()
        .to_string()
    )

    print("\nPlatform:")

    print(
        df["platform"]
        .value_counts()
        .to_string()
    )

    print("\nMedia Type:")

    print(
        df["media_type"]
        .value_counts()
        .to_string()
    )

    print("\nLanguage:")

    print(
        df["language"]
        .value_counts()
        .to_string()
    )

    print("\nSeverity:")

    print(
        df["severity"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # REPORT PATHS
    # --------------------------------------------------------

    print("\nReports:")

    print(
        f"  {VALIDATION_REPORT}"
    )

    print(
        f"  {CATEGORY_REPORT}"
    )

    print(
        f"  {SEVERITY_REPORT}"
    )

    print(
        f"  {TARGET_REPORT}"
    )

    print(
        f"  {LANGUAGE_REPORT}"
    )

    print(
        f"  {DATA_SOURCE_REPORT}"
    )

    print(
        f"  {INTENT_REPORT}"
    )

    print(
        f"  {CONTEXT_REPORT}"
    )

    print(
        f"  {ENVIRONMENT_REPORT}"
    )

    print(
        f"  {PLATFORM_REPORT}"
    )

    print(
        f"  {MEDIA_TYPE_REPORT}"
    )

    print(
        f"  {CONTENT_CATEGORY_REPORT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()