import os
import re
import unicodedata

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_robust_multilingual.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_robust_multilingual_final.csv"
)

REPORT_PATH = (
    "reports/"
    "robust_cleaning_summary.csv"
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

VALID_DATA_SOURCES = {
    "synthetic_tanglish",
    "synthetic_english",
    "synthetic_robustness",
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


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Canonical text normalization for Tanglish and English.

    Important:
    - Keeps word boundaries.
    - Does NOT remove spaces between words.
    - Space-free representation will be created later
      during TF-IDF feature engineering.
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
    # Convert multiple whitespace characters into one space
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
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 2 — ROBUST MULTILINGUAL DATASET CLEANING")
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
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8"
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
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "id",
        "text",
        "language",
        "cyberbullying",
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
            f"Missing required columns: {missing_columns}"
        )

    print("\nRequired columns: OK")

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    duplicate_rows = (
        df.duplicated()
        .sum()
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

    df["text_before_cleaning"] = df["text"]

    df["text"] = (
        df["text"]
        .apply(clean_text)
    )

    empty_after_cleaning = (
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

    df["language"] = (
        df["language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_languages = (
        ~df["language"].isin(
            VALID_LANGUAGES
        )
    )

    invalid_language_count = (
        invalid_languages.sum()
    )

    if invalid_language_count > 0:

        print(
            f"\nWarning: "
            f"{invalid_language_count} "
            f"invalid language values found."
        )

        print(
            df.loc[
                invalid_languages,
                "language"
            ].value_counts()
        )

    # --------------------------------------------------------
    # Normalize data source
    # --------------------------------------------------------

    df["data_source"] = (
        df["data_source"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_data_sources = (
        ~df["data_source"].isin(
            VALID_DATA_SOURCES
        )
    )

    invalid_data_source_count = (
        invalid_data_sources.sum()
    )

    if invalid_data_source_count > 0:

        print(
            f"\nWarning: "
            f"{invalid_data_source_count} "
            f"invalid data_source values found."
        )

        print(
            df.loc[
                invalid_data_sources,
                "data_source"
            ].value_counts()
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

        invalid_values = (
            original_values.isna()
            & df[column].notna()
        ).sum()

        invalid_binary_count += (
            invalid_values
        )

        df[column] = (
            original_values
            .fillna(0)
            .astype(int)
        )

        # ----------------------------------------------------
        # Make sure binary values are only 0 or 1
        # ----------------------------------------------------

        invalid_range = (
            ~df[column].isin([0, 1])
        ).sum()

        if invalid_range > 0:

            invalid_binary_count += (
                invalid_range
            )

            print(
                f"Warning: {column} contains "
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

    invalid_target_count = (
        invalid_target_types.sum()
    )

    if invalid_target_count > 0:

        print(
            f"\nWarning: "
            f"{invalid_target_count} "
            f"invalid target types found."
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

    invalid_severity_count = (
        invalid_severity.sum()
    )

    if invalid_severity_count > 0:

        print(
            f"\nWarning: "
            f"{invalid_severity_count} "
            f"invalid severity values found."
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
    # Remove temporary column
    # --------------------------------------------------------

    df.drop(
        columns=[
            "text_before_cleaning"
        ],
        inplace=True
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
    print("ROBUST DATASET CLEANING SUMMARY")
    print("=" * 75)

    print(
        f"Original rows                : "
        f"{original_rows}"
    )

    print(
        f"Final rows                   : "
        f"{len(df)}"
    )

    print(
        f"Duplicate rows removed       : "
        f"{duplicate_rows}"
    )

    print(
        f"Empty texts removed          : "
        f"{empty_after_cleaning}"
    )

    print(
        f"Invalid language values      : "
        f"{invalid_language_count}"
    )

    print(
        f"Invalid data source values   : "
        f"{invalid_data_source_count}"
    )

    print(
        f"Invalid binary values        : "
        f"{invalid_binary_count}"
    )

    print(
        f"Invalid target types         : "
        f"{invalid_target_count}"
    )

    print(
        f"Invalid severity values      : "
        f"{invalid_severity_count}"
    )

    print(
        f"Final columns                : "
        f"{len(df.columns)}"
    )

    print("\nLanguage distribution:")

    print(
        df["language"]
        .value_counts()
        .to_string()
    )

    print("\nData source distribution:")

    print(
        df["data_source"]
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

    print(
        "\nStep 2 completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()