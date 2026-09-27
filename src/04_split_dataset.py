import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_robust_multilingual_final.csv"
)

OUTPUT_DIR = (
    "data/processed/"
    "splits_robust"
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
    print("STEP 4 — ROBUST MULTILINGUAL DATASET SPLITTING")
    print("=" * 75)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset:")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Create stratification label
    #
    # We preserve:
    # 1. Cyberbullying distribution
    # 2. Language distribution
    # --------------------------------------------------------

    df["stratify_label"] = (
        df["cyberbullying"].astype(str)
        + "_"
        + df["language"].astype(str)
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
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["stratify_label"]
    )

    # --------------------------------------------------------
    # Validation + Test
    #
    # Temporary 20% is divided equally:
    #
    # 10% validation
    # 10% test
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

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Save splits
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

    train_df.to_csv(
        train_path,
        index=False
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Print dataset sizes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Cyberbullying distribution
    # --------------------------------------------------------

    print("\nCyberbullying distribution:")

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

    # --------------------------------------------------------
    # Language distribution
    # --------------------------------------------------------

    print("\nLanguage distribution:")

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

    # --------------------------------------------------------
    # Language × Cyberbullying
    # --------------------------------------------------------

    print("\nLanguage × Cyberbullying:")

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

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(df):
        raise ValueError(
            "Split row count does not match original dataset."
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
    # Output paths
    # --------------------------------------------------------

    print("\nOutput files:")

    print(train_path)
    print(validation_path)
    print(test_path)

    print("\nStep 4 completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()