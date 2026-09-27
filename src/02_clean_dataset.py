import os
import re
import unicodedata

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    "data/raw/"
    "cyberbullying_25000_realworld_multilingual_v3.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_25000_realworld_multilingual_v3.csv"
)

REPORT_PATH = (
    "reports/"
    "realworld_cleaning_summary.csv"
)


# ============================================================
# CATEGORY COLUMNS
# ============================================================

CATEGORY_COLUMNS = [
    "insult",
    "threat",
    "violence",
    "hate",
    "sexual_harassment",
    "sexual_abuse",
    "profanity",
]


# ============================================================
# BINARY COLUMNS
# ============================================================

BINARY_COLUMNS = [
    "cyberbullying",
    *CATEGORY_COLUMNS,
]


# ============================================================
# VALID VALUES
# ============================================================

VALID_LANGUAGES = {
    "tanglish",
    "english",
}


VALID_CONTEXT_TYPES = {
    "social_media",
    "non_social_realworld",
}


VALID_ENVIRONMENTS = {
    "social_media",
    "gaming_chat",
    "personal_chat",
    "messaging",
    "workplace",
    "school_college",
    "review_feedback",
    "public_forum",
    "email",
    "community_chat",
}


VALID_PLATFORMS = {
    "instagram",
    "youtube",
    "facebook",
    "socialmedia",
    "nonsocial",
}


VALID_MEDIA_TYPES = {
    "post",
    "reel",
    "story",
    "photo",
    "video",
    "caption",
    "comment",
    "message",
}


VALID_INTENTS = {
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
    "neutral",
}


VALID_CONTENT_CATEGORIES = {
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
    "sexual_abuse",
}


VALID_SEVERITY = {
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


VALID_DATA_SOURCES = {
    "synthetic_curated_realworld_v3",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Canonical text normalization for English and Tanglish.

    Important:
    - Keeps word boundaries.
    - Keeps meaningful punctuation.
    - Does NOT remove offensive words.
    - Does NOT remove Tanglish slang.
    - Does NOT remove social-media terminology.
    - Does NOT remove spaces between words.
    - Space-free representation will be created later
      during feature engineering.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # --------------------------------------------------------
    # Remove leading and trailing spaces
    # --------------------------------------------------------

    text = text.strip()

    # --------------------------------------------------------
    # Normalize multiple whitespace characters
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize repeated punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"!{2,}",
        "!",
        text
    )

    text = re.sub(
        r"\?{2,}",
        "?",
        text
    )

    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    return text


# ============================================================
# STRING NORMALIZATION
# ============================================================

def normalize_string_column(
    df,
    column
):

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# ============================================================
# VALIDATE STRING COLUMN
# ============================================================

def validate_string_column(
    df,
    column,
    valid_values
):

    invalid_mask = (
        ~df[column].isin(valid_values)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:

        print(
            f"\nWarning: {invalid_count} "
            f"invalid {column} values found."
        )

        print(
            df.loc[
                invalid_mask,
                column
            ]
            .value_counts()
            .to_string()
        )

    return invalid_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 2 — REAL-WORLD MULTILINGUAL DATASET CLEANING")
    print("=" * 75)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"\nInput dataset not found:\n"
            f"{INPUT_PATH}\n\n"
            "Place the 25K CSV inside data/raw/"
        )

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig"
    )

    print("\nOriginal dataset:")

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    original_rows = len(df)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

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
        *CATEGORY_COLUMNS,
        "severity",
        "reason",
        "data_source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    print(
        "\nRequired columns: OK"
    )

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if duplicate_rows > 0:

        df = (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

    print(
        f"\nDuplicate rows removed : "
        f"{duplicate_rows}"
    )

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .apply(clean_text)
    )

    empty_after_cleaning = int(
        df["text"]
        .str.len()
        .eq(0)
        .sum()
    )

    print(
        f"Empty texts after cleaning : "
        f"{empty_after_cleaning}"
    )

    # --------------------------------------------------------
    # Remove empty texts
    # --------------------------------------------------------

    if empty_after_cleaning > 0:

        df = df[
            df["text"].str.len() > 0
        ].copy()

    # --------------------------------------------------------
    # Clean ID
    # --------------------------------------------------------

    df["id"] = (
        df["id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Normalize language
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "language"
    )

    invalid_language_count = (
        validate_string_column(
            df,
            "language",
            VALID_LANGUAGES
        )
    )

    # --------------------------------------------------------
    # Normalize context type
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "context_type"
    )

    invalid_context_count = (
        validate_string_column(
            df,
            "context_type",
            VALID_CONTEXT_TYPES
        )
    )

    # --------------------------------------------------------
    # Normalize environment
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "environment"
    )

    invalid_environment_count = (
        validate_string_column(
            df,
            "environment",
            VALID_ENVIRONMENTS
        )
    )

    # --------------------------------------------------------
    # Normalize platform
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "platform"
    )

    invalid_platform_count = (
        validate_string_column(
            df,
            "platform",
            VALID_PLATFORMS
        )
    )

    # --------------------------------------------------------
    # Normalize media type
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "media_type"
    )

    invalid_media_type_count = (
        validate_string_column(
            df,
            "media_type",
            VALID_MEDIA_TYPES
        )
    )

    # --------------------------------------------------------
    # Normalize intent
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "intent"
    )

    invalid_intent_count = (
        validate_string_column(
            df,
            "intent",
            VALID_INTENTS
        )
    )

    # --------------------------------------------------------
    # Normalize content category
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "content_category"
    )

    invalid_content_category_count = (
        validate_string_column(
            df,
            "content_category",
            VALID_CONTENT_CATEGORIES
        )
    )

    # --------------------------------------------------------
    # Normalize data source
    # --------------------------------------------------------

    df = normalize_string_column(
        df,
        "data_source"
    )

    invalid_data_source_count = (
        validate_string_column(
            df,
            "data_source",
            VALID_DATA_SOURCES
        )
    )

    # --------------------------------------------------------
    # Convert binary labels to integers
    # --------------------------------------------------------

    invalid_binary_count = 0

    for column in BINARY_COLUMNS:

        original_values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_nan_values = (
            original_values.isna()
            & df[column].notna()
        ).sum()

        invalid_binary_count += int(
            invalid_nan_values
        )

        df[column] = (
            original_values
            .fillna(0)
            .astype(int)
        )

        # ----------------------------------------------------
        # Ensure values are only 0 or 1
        # ----------------------------------------------------

        invalid_range = (
            ~df[column].isin([0, 1])
        ).sum()

        if invalid_range > 0:

            invalid_binary_count += int(
                invalid_range
            )

            print(
                f"\nWarning: {column} contains "
                f"{invalid_range} non-binary values."
            )

            df[column] = (
                df[column]
                .apply(
                    lambda value:
                    1 if value >= 1 else 0
                )
            )

    # --------------------------------------------------------
    # Normalize target type
    # --------------------------------------------------------

    df["target_type"] = (
        df["target_type"]
        .fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_target_types = (
        ~df["target_type"].isin(
            VALID_TARGET_TYPES
        )
    )

    invalid_target_count = int(
        invalid_target_types.sum()
    )

    if invalid_target_count > 0:

        print(
            f"\nWarning: "
            f"{invalid_target_count} "
            f"invalid target types found."
        )

        print(
            df.loc[
                invalid_target_types,
                "target_type"
            ]
            .value_counts()
            .to_string()
        )

        df.loc[
            invalid_target_types,
            "target_type"
        ] = "other"

    # --------------------------------------------------------
    # Normalize severity
    # --------------------------------------------------------

    df["severity"] = (
        df["severity"]
        .fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_severity = (
        ~df["severity"].isin(
            VALID_SEVERITY
        )
    )

    invalid_severity_count = int(
        invalid_severity.sum()
    )

    if invalid_severity_count > 0:

        print(
            f"\nWarning: "
            f"{invalid_severity_count} "
            f"invalid severity values found."
        )

        print(
            df.loc[
                invalid_severity,
                "severity"
            ]
            .value_counts()
            .to_string()
        )

        df.loc[
            invalid_severity,
            "severity"
        ] = "none"

    # --------------------------------------------------------
    # Clean reason
    # --------------------------------------------------------

    df["reason"] = (
        df["reason"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove accidental unnamed columns
    # --------------------------------------------------------

    unnamed_columns = [
        column
        for column in df.columns
        if column.startswith("Unnamed:")
    ]

    if unnamed_columns:

        df.drop(
            columns=unnamed_columns,
            inplace=True
        )

        print(
            "\nUnnamed columns removed:"
        )

        for column in unnamed_columns:
            print(
                f"  - {column}"
            )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Cleaning summary
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "metric": [

            "original_rows",

            "final_rows",

            "duplicate_rows_removed",

            "empty_texts_removed",

            "invalid_language_values",

            "invalid_context_type_values",

            "invalid_environment_values",

            "invalid_platform_values",

            "invalid_media_type_values",

            "invalid_intent_values",

            "invalid_content_category_values",

            "invalid_data_source_values",

            "invalid_binary_values",

            "invalid_target_types",

            "invalid_severity_values",

            "final_columns",

        ],

        "value": [

            original_rows,

            len(df),

            duplicate_rows,

            empty_after_cleaning,

            invalid_language_count,

            invalid_context_count,

            invalid_environment_count,

            invalid_platform_count,

            invalid_media_type_count,

            invalid_intent_count,

            invalid_content_category_count,

            invalid_data_source_count,

            invalid_binary_count,

            invalid_target_count,

            invalid_severity_count,

            len(df.columns),

        ],

    })

    summary.to_csv(
        REPORT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("REAL-WORLD DATASET CLEANING SUMMARY")
    print("=" * 75)

    print(
        f"Original rows                  : "
        f"{original_rows}"
    )

    print(
        f"Final rows                     : "
        f"{len(df)}"
    )

    print(
        f"Duplicate rows removed         : "
        f"{duplicate_rows}"
    )

    print(
        f"Empty texts removed            : "
        f"{empty_after_cleaning}"
    )

    print(
        f"Invalid language values        : "
        f"{invalid_language_count}"
    )

    print(
        f"Invalid context values         : "
        f"{invalid_context_count}"
    )

    print(
        f"Invalid environment values     : "
        f"{invalid_environment_count}"
    )

    print(
        f"Invalid platform values        : "
        f"{invalid_platform_count}"
    )

    print(
        f"Invalid media type values      : "
        f"{invalid_media_type_count}"
    )

    print(
        f"Invalid intent values          : "
        f"{invalid_intent_count}"
    )

    print(
        f"Invalid content category       : "
        f"{invalid_content_category_count}"
    )

    print(
        f"Invalid data source values     : "
        f"{invalid_data_source_count}"
    )

    print(
        f"Invalid binary values          : "
        f"{invalid_binary_count}"
    )

    print(
        f"Invalid target types           : "
        f"{invalid_target_count}"
    )

    print(
        f"Invalid severity values        : "
        f"{invalid_severity_count}"
    )

    print(
        f"Final columns                  : "
        f"{len(df.columns)}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\nLanguage distribution:")

    print(
        df["language"]
        .value_counts()
        .to_string()
    )

    print("\nContext type distribution:")

    print(
        df["context_type"]
        .value_counts()
        .to_string()
    )

    print("\nEnvironment distribution:")

    print(
        df["environment"]
        .value_counts()
        .to_string()
    )

    print("\nPlatform distribution:")

    print(
        df["platform"]
        .value_counts()
        .to_string()
    )

    print("\nIntent distribution:")

    print(
        df["intent"]
        .value_counts()
        .to_string()
    )

    print("\nContent category distribution:")

    print(
        df["content_category"]
        .value_counts()
        .to_string()
    )

    print("\nCyberbullying distribution:")

    print(
        df["cyberbullying"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSeverity distribution:")

    print(
        df["severity"]
        .value_counts()
        .to_string()
    )

    print("\nTarget type distribution:")

    print(
        df["target_type"]
        .value_counts()
        .to_string()
    )

    print("\nProcessed dataset:")
    print(
        OUTPUT_PATH
    )

    print("\nCleaning report:")
    print(
        REPORT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()