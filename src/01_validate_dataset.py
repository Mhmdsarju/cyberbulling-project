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
# FILES
# ============================================================

INPUT_FILE = (
    RAW_DIR /
    "cyberbullying_multilingual_9000.csv"
)

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


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_ROWS = 9000

EXPECTED_LANGUAGES = {
    "tanglish",
    "english"
}

EXPECTED_DATA_SOURCES = {
    "synthetic_tanglish",
    "synthetic_english"
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
            "Please place the dataset inside data/raw/"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8"
    )

    return df


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

def check_columns(df):

    required_columns = [
        "id",
        "text",
        "language",
        "cyberbullying",
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

        print("\nMissing columns:")

        for column in missing:
            print(f"  - {column}")

        raise ValueError(
            "\nDataset structure is invalid."
        )

    print("\nRequired columns: OK")


# ============================================================
# CHECK EMPTY VALUES
# ============================================================

def check_missing_values(df):

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print("  No missing values.")

    else:

        print(missing.to_string())

    return missing


# ============================================================
# CHECK DUPLICATES
# ============================================================

def check_duplicates(df):

    duplicate_ids = (
        df["id"]
        .duplicated()
        .sum()
    )

    duplicate_texts = (
        df["text"]
        .duplicated()
        .sum()
    )

    print("\nDuplicates:")

    print(
        f"  Duplicate IDs   : {duplicate_ids}"
    )

    print(
        f"  Duplicate texts : {duplicate_texts}"
    )

    return (
        duplicate_ids,
        duplicate_texts
    )


# ============================================================
# CHECK ROW COUNT
# ============================================================

def check_row_count(df):

    actual_rows = len(df)

    print("\nRow count:")

    print(
        f"  Expected : {EXPECTED_ROWS}"
    )

    print(
        f"  Actual   : {actual_rows}"
    )

    if actual_rows == EXPECTED_ROWS:

        print("  Row count: OK")

        return True

    print("  Row count: WARNING")

    return False


# ============================================================
# CHECK CYBERBULLYING
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

    invalid = values - {0, 1}

    print("\nCyberbullying labels:")

    print(
        df["cyberbullying"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if invalid:

        print(
            f"\nInvalid cyberbullying values: {invalid}"
        )

    else:

        print(
            "\nCyberbullying values: OK"
        )

    return invalid


# ============================================================
# CHECK LANGUAGE
# ============================================================

def check_language(df):

    values = set(
        df["language"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    invalid = values - EXPECTED_LANGUAGES

    print("\nLanguage distribution:")

    print(
        df["language"]
        .value_counts()
        .to_string()
    )

    if invalid:

        print(
            f"\nInvalid language values: {invalid}"
        )

    else:

        print(
            "\nLanguage values: OK"
        )

    return invalid


# ============================================================
# CHECK DATA SOURCE
# ============================================================

def check_data_source(df):

    values = set(
        df["data_source"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    invalid = values - EXPECTED_DATA_SOURCES

    print("\nData source distribution:")

    print(
        df["data_source"]
        .value_counts()
        .to_string()
    )

    if invalid:

        print(
            f"\nInvalid data_source values: {invalid}"
        )

    else:

        print(
            "\nData source values: OK"
        )

    return invalid


# ============================================================
# CHECK CATEGORIES
# ============================================================

def check_categories(df):

    print("\nCategory validation:")

    invalid_values = {}

    for column in CATEGORY_COLUMNS:

        values = set(
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        invalid = values - {0, 1}

        if invalid:

            invalid_values[column] = invalid

        print(
            f"{column:20s}: "
            f"positive={int(df[column].sum())}"
        )

    if invalid_values:

        print("\nInvalid category values:")

        for column, values in invalid_values.items():

            print(
                f"  {column}: {values}"
            )

    else:

        print(
            "\nAll category values are valid 0/1."
        )

    return invalid_values


# ============================================================
# CHECK SEVERITY
# ============================================================

def check_severity(df):

    values = set(
        df["severity"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    invalid = values - EXPECTED_SEVERITY

    print("\nSeverity distribution:")

    print(
        df["severity"]
        .value_counts()
        .to_string()
    )

    if invalid:

        print(
            f"\nInvalid severity values: {invalid}"
        )

    else:

        print(
            "\nSeverity values: OK"
        )

    return invalid


# ============================================================
# CHECK TARGET TYPE
# ============================================================

def check_target_type(df):

    values = set(
        df["target_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    invalid = values - EXPECTED_TARGET_TYPES

    print("\nTarget type distribution:")

    print(
        df["target_type"]
        .value_counts()
        .to_string()
    )

    if invalid:

        print(
            f"\nInvalid target types: {invalid}"
        )

    else:

        print(
            "\nTarget type values: OK"
        )

    return invalid


# ============================================================
# CHECK TEXT
# ============================================================

def check_text(df):

    text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_count = (
        text == ""
    ).sum()

    lengths = text.str.len()

    print("\nText statistics:")

    print(
        f"  Empty texts : {empty_count}"
    )

    print(
        f"  Minimum     : {lengths.min()}"
    )

    print(
        f"  Maximum     : {lengths.max()}"
    )

    print(
        f"  Mean        : {lengths.mean():.2f}"
    )

    print(
        f"  Median      : {lengths.median():.2f}"
    )

    return {
        "empty": empty_count,
        "min": lengths.min(),
        "max": lengths.max(),
        "mean": lengths.mean(),
        "median": lengths.median()
    }


# ============================================================
# SAVE DISTRIBUTION REPORTS
# ============================================================

def save_reports(df):

    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    category_rows = []

    for column in CATEGORY_COLUMNS:

        positive = int(
            df[column].sum()
        )

        negative = len(df) - positive

        category_rows.append({
            "category": column,
            "positive": positive,
            "negative": negative,
            "positive_percentage":
                round(
                    positive / len(df) * 100,
                    2
                )
        })

    category_df = pd.DataFrame(
        category_rows
    )

    category_df.to_csv(
        CATEGORY_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_df = (
        df["severity"]
        .value_counts()
        .rename_axis("severity")
        .reset_index(name="count")
    )

    severity_df["percentage"] = (
        severity_df["count"]
        / len(df)
        * 100
    ).round(2)

    severity_df.to_csv(
        SEVERITY_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # Target type
    # --------------------------------------------------------

    target_df = (
        df["target_type"]
        .value_counts()
        .rename_axis("target_type")
        .reset_index(name="count")
    )

    target_df["percentage"] = (
        target_df["count"]
        / len(df)
        * 100
    ).round(2)

    target_df.to_csv(
        TARGET_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language_df = (
        df["language"]
        .value_counts()
        .rename_axis("language")
        .reset_index(name="count")
    )

    language_df["percentage"] = (
        language_df["count"]
        / len(df)
        * 100
    ).round(2)

    language_df.to_csv(
        LANGUAGE_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # Data source
    # --------------------------------------------------------

    data_source_df = (
        df["data_source"]
        .value_counts()
        .rename_axis("data_source")
        .reset_index(name="count")
    )

    data_source_df["percentage"] = (
        data_source_df["count"]
        / len(df)
        * 100
    ).round(2)

    data_source_df.to_csv(
        DATA_SOURCE_REPORT,
        index=False
    )

    return (
        category_df,
        severity_df,
        target_df,
        language_df,
        data_source_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 1 — ROBUST MULTILINGUAL DATASET VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print(
        "\nDataset loaded successfully."
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    check_columns(df)

    row_count_ok = check_row_count(df)

    missing = check_missing_values(df)

    duplicate_ids, duplicate_texts = (
        check_duplicates(df)
    )

    invalid_binary = (
        check_cyberbullying(df)
    )

    invalid_language = (
        check_language(df)
    )

    invalid_data_source = (
        check_data_source(df)
    )

    invalid_categories = (
        check_categories(df)
    )

    invalid_severity = (
        check_severity(df)
    )

    invalid_target = (
        check_target_type(df)
    )

    text_stats = check_text(df)

    # --------------------------------------------------------
    # Save distribution reports
    # --------------------------------------------------------

    (
        category_df,
        severity_df,
        target_df,
        language_df,
        data_source_df
    ) = save_reports(df)

    # --------------------------------------------------------
    # Overall validation summary
    # --------------------------------------------------------

    validation_rows = [

        {
            "check": "total_rows",
            "value": len(df),
            "status":
                "PASS"
                if row_count_ok
                else "WARNING"
        },

        {
            "check": "total_columns",
            "value": len(df.columns),
            "status": "PASS"
        },

        {
            "check": "missing_values",
            "value": int(missing.sum()),
            "status":
                "PASS"
                if missing.sum() == 0
                else "WARNING"
        },

        {
            "check": "duplicate_ids",
            "value": duplicate_ids,
            "status":
                "PASS"
                if duplicate_ids == 0
                else "WARNING"
        },

        {
            "check": "duplicate_texts",
            "value": duplicate_texts,
            "status":
                "PASS"
                if duplicate_texts == 0
                else "WARNING"
        },

        {
            "check": "invalid_cyberbullying_values",
            "value": len(invalid_binary),
            "status":
                "PASS"
                if len(invalid_binary) == 0
                else "FAIL"
        },

        {
            "check": "invalid_language_values",
            "value": len(invalid_language),
            "status":
                "PASS"
                if len(invalid_language) == 0
                else "FAIL"
        },

        {
            "check": "invalid_data_source_values",
            "value": len(invalid_data_source),
            "status":
                "PASS"
                if len(invalid_data_source) == 0
                else "FAIL"
        },

        {
            "check": "invalid_category_values",
            "value": len(invalid_categories),
            "status":
                "PASS"
                if len(invalid_categories) == 0
                else "FAIL"
        },

        {
            "check": "invalid_severity_values",
            "value": len(invalid_severity),
            "status":
                "PASS"
                if len(invalid_severity) == 0
                else "FAIL"
        },

        {
            "check": "invalid_target_type_values",
            "value": len(invalid_target),
            "status":
                "PASS"
                if len(invalid_target) == 0
                else "FAIL"
        },

        {
            "check": "empty_texts",
            "value": text_stats["empty"],
            "status":
                "PASS"
                if text_stats["empty"] == 0
                else "WARNING"
        }
    ]

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_df.to_csv(
        VALIDATION_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("VALIDATION SUMMARY")
    print("=" * 75)

    print(
        validation_df.to_string(
            index=False
        )
    )

    print("\nReports created:")

    print(
        VALIDATION_REPORT
    )

    print(
        CATEGORY_REPORT
    )

    print(
        SEVERITY_REPORT
    )

    print(
        TARGET_REPORT
    )

    print(
        LANGUAGE_REPORT
    )

    print(
        DATA_SOURCE_REPORT
    )

    print(
        "\nStep 1 completed successfully."
    )


if __name__ == "__main__":
    main()