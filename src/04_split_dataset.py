import os

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_25000_realworld_multilingual_v3.csv"
)

OUTPUT_DIR = (
    "data/processed/"
    "splits_realworld"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 4 — REAL-WORLD MULTILINGUAL DATASET SPLITTING")
    print("=" * 75)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig"
    )

    print("\nDataset:")

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate required columns
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

    # --------------------------------------------------------
    # Validate dataset size
    # --------------------------------------------------------

    if len(df) != 25000:

        print(
            f"\nWarning: expected 25000 rows, "
            f"found {len(df)} rows."
        )

    # --------------------------------------------------------
    # Create stratification label
    #
    # Preserve:
    # 1. Cyberbullying distribution
    # 2. Language distribution
    #
    # This gives four groups:
    #
    # english + non-cyberbullying
    # english + cyberbullying
    # tanglish + non-cyberbullying
    # tanglish + cyberbullying
    # --------------------------------------------------------

    df["stratify_label"] = (
        df["cyberbullying"]
        .astype(str)
        + "_"
        + df["language"]
        .astype(str)
    )

    print("\nOriginal language × cyberbullying:")

    print(
        pd.crosstab(
            df["language"],
            df["cyberbullying"]
        )
    )

    # --------------------------------------------------------
    # Train + Temporary split
    #
    # 80% train
    # 20% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VALIDATION_SIZE
            + TEST_SIZE
        ),
        random_state=RANDOM_STATE,
        stratify=df["stratify_label"]
    )

    # --------------------------------------------------------
    # Validation + Test
    #
    # Temporary 20%:
    #
    # 50% validation = 10% total
    # 50% test       = 10% total
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["stratify_label"]
    )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    train_df = train_df.drop(
        columns=["stratify_label"]
    )

    validation_df = validation_df.drop(
        columns=["stratify_label"]
    )

    test_df = test_df.drop(
        columns=["stratify_label"]
    )

    # --------------------------------------------------------
    # Reset indexes
    # --------------------------------------------------------

    train_df = (
        train_df
        .reset_index(drop=True)
    )

    validation_df = (
        validation_df
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save split paths
    # --------------------------------------------------------

    train_path = os.path.join(
        OUTPUT_DIR,
        "train.csv"
    )

    validation_path = os.path.join(
        OUTPUT_DIR,
        "validation.csv"
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        "test.csv"
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    train_df.to_csv(
        train_path,
        index=False,
        encoding="utf-8"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("SPLIT SUMMARY")
    print("=" * 75)

    print(
        f"Total dataset       : {len(df)}"
    )

    print(
        f"Training set        : {len(train_df)} "
        f"({len(train_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Validation set      : {len(validation_df)} "
        f"({len(validation_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Test set            : {len(test_df)} "
        f"({len(test_df) / len(df) * 100:.2f}%)"
    )

    # ========================================================
    # CYBERBULLYING DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("CYBERBULLYING DISTRIBUTION")
    print("-" * 75)

    print("\nTrain:")

    print(
        train_df["cyberbullying"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation:")

    print(
        validation_df["cyberbullying"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTest:")

    print(
        test_df["cyberbullying"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # LANGUAGE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("LANGUAGE DISTRIBUTION")
    print("-" * 75)

    print("\nTrain:")

    print(
        train_df["language"]
        .value_counts()
        .to_string()
    )

    print("\nValidation:")

    print(
        validation_df["language"]
        .value_counts()
        .to_string()
    )

    print("\nTest:")

    print(
        test_df["language"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # LANGUAGE × CYBERBULLYING
    # ========================================================

    print("\n" + "-" * 75)
    print("LANGUAGE × CYBERBULLYING")
    print("-" * 75)

    print("\nTrain:")

    print(
        pd.crosstab(
            train_df["language"],
            train_df["cyberbullying"]
        )
    )

    print("\nValidation:")

    print(
        pd.crosstab(
            validation_df["language"],
            validation_df["cyberbullying"]
        )
    )

    print("\nTest:")

    print(
        pd.crosstab(
            test_df["language"],
            test_df["cyberbullying"]
        )
    )

    # ========================================================
    # CONTEXT TYPE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("CONTEXT TYPE DISTRIBUTION")
    print("-" * 75)

    print("\nTrain:")

    print(
        train_df["context_type"]
        .value_counts()
        .to_string()
    )

    print("\nValidation:")

    print(
        validation_df["context_type"]
        .value_counts()
        .to_string()
    )

    print("\nTest:")

    print(
        test_df["context_type"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # CONTEXT × CYBERBULLYING
    # ========================================================

    print("\n" + "-" * 75)
    print("CONTEXT × CYBERBULLYING")
    print("-" * 75)

    print("\nTrain:")

    print(
        pd.crosstab(
            train_df["context_type"],
            train_df["cyberbullying"]
        )
    )

    print("\nValidation:")

    print(
        pd.crosstab(
            validation_df["context_type"],
            validation_df["cyberbullying"]
        )
    )

    print("\nTest:")

    print(
        pd.crosstab(
            test_df["context_type"],
            test_df["cyberbullying"]
        )
    )

    # ========================================================
    # PLATFORM DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("PLATFORM DISTRIBUTION")
    print("-" * 75)

    print("\nTrain:")

    print(
        train_df["platform"]
        .value_counts()
        .to_string()
    )

    print("\nValidation:")

    print(
        validation_df["platform"]
        .value_counts()
        .to_string()
    )

    print("\nTest:")

    print(
        test_df["platform"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # ENVIRONMENT DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("ENVIRONMENT DISTRIBUTION")
    print("-" * 75)

    print("\nTrain:")

    print(
        train_df["environment"]
        .value_counts()
        .to_string()
    )

    print("\nValidation:")

    print(
        validation_df["environment"]
        .value_counts()
        .to_string()
    )

    print("\nTest:")

    print(
        test_df["environment"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(df):

        raise ValueError(
            "Split row count does not match "
            "original dataset."
        )

    if (
        len(train_df) == 0
        or len(validation_df) == 0
        or len(test_df) == 0
    ):

        raise ValueError(
            "One of the dataset splits is empty."
        )

    # --------------------------------------------------------
    # Check duplicate IDs across splits
    # --------------------------------------------------------

    train_ids = set(
        train_df["id"]
    )

    validation_ids = set(
        validation_df["id"]
    )

    test_ids = set(
        test_df["id"]
    )

    if train_ids & validation_ids:

        raise ValueError(
            "Duplicate IDs found between "
            "train and validation."
        )

    if train_ids & test_ids:

        raise ValueError(
            "Duplicate IDs found between "
            "train and test."
        )

    if validation_ids & test_ids:

        raise ValueError(
            "Duplicate IDs found between "
            "validation and test."
        )

    # --------------------------------------------------------
    # Check text overlap across splits
    # --------------------------------------------------------

    train_texts = set(
        train_df["text"]
    )

    validation_texts = set(
        validation_df["text"]
    )

    test_texts = set(
        test_df["text"]
    )

    if train_texts & validation_texts:

        raise ValueError(
            "Duplicate texts found between "
            "train and validation."
        )

    if train_texts & test_texts:

        raise ValueError(
            "Duplicate texts found between "
            "train and test."
        )

    if validation_texts & test_texts:

        raise ValueError(
            "Duplicate texts found between "
            "validation and test."
        )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("\nOutput files:")

    print(
        f"Train      : {train_path}"
    )

    print(
        f"Validation : {validation_path}"
    )

    print(
        f"Test       : {test_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()