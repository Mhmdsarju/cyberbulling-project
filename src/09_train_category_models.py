import os
import joblib
import pandas as pd

from scipy.sparse import load_npz
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = (
    "data/processed/"
    "splits_robust/train.csv"
)

VALIDATION_PATH = (
    "data/processed/"
    "splits_robust/validation.csv"
)

TEST_PATH = (
    "data/processed/"
    "splits_robust/test.csv"
)

FEATURE_DIR = (
    "data/processed/"
    "features_robust"
)

MODEL_DIR = (
    "models/category_models"
)

REPORT_DIR = (
    "reports/category_models"
)

CATEGORY_COLUMNS = [
    "insult",
    "threat",
    "violence",
    "hate",
    "sexual_harassment",
    "sexual_abuse",
    "profanity",
]

MODEL_PATH = (
    "models/category_models/"
    "category_multilabel_svm_robust_multilingual.joblib"
)

VALIDATION_REPORT_PATH = (
    "reports/category_models/"
    "robust_category_validation_results.csv"
)

TEST_REPORT_PATH = (
    "reports/category_models/"
    "robust_category_test_results.csv"
)

TEST_SUMMARY_PATH = (
    "reports/category_models/"
    "robust_category_test_summary.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    train_path = os.path.join(
        FEATURE_DIR,
        "X_train_tfidf_robust.npz"
    )

    validation_path = os.path.join(
        FEATURE_DIR,
        "X_validation_tfidf_robust.npz"
    )

    test_path = os.path.join(
        FEATURE_DIR,
        "X_test_tfidf_robust.npz"
    )

    for path in [
        train_path,
        validation_path,
        test_path,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Feature file not found: {path}"
            )

    X_train = load_npz(
        train_path
    )

    X_validation = load_npz(
        validation_path
    )

    X_test = load_npz(
        test_path
    )

    return (
        X_train,
        X_validation,
        X_test
    )


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    for path in [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    X_train,
    X_validation,
    X_test
):

    print(
        "\nFeature validation:"
    )

    print(
        f"Train      : {X_train.shape}"
    )

    print(
        f"Validation : {X_validation.shape}"
    )

    print(
        f"Test       : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Feature dimensions
    # --------------------------------------------------------

    if (
        X_train.shape[1]
        != X_validation.shape[1]
        or
        X_train.shape[1]
        != X_test.shape[1]
    ):

        raise ValueError(
            "Train, validation and test "
            "feature dimensions do not match."
        )

    print(
        f"Feature dimension : "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------

    if (
        X_train.shape[0]
        != 7220
    ):

        raise ValueError(
            "Unexpected training feature row count."
        )

    if (
        X_validation.shape[0]
        != 902
    ):

        raise ValueError(
            "Unexpected validation feature row count."
        )

    if (
        X_test.shape[0]
        != 903
    ):

        raise ValueError(
            "Unexpected test feature row count."
        )

    print(
        "Feature dimensions and row counts are valid."
    )


# ============================================================
# VALIDATE LABELS
# ============================================================

def validate_labels(
    train_df,
    validation_df,
    test_df,
    y_train,
    y_validation,
    y_test
):

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------

    if len(train_df) != len(y_train):

        raise ValueError(
            "Training dataset and labels do not match."
        )

    if len(validation_df) != len(
        y_validation
    ):

        raise ValueError(
            "Validation dataset and labels do not match."
        )

    if len(test_df) != len(y_test):

        raise ValueError(
            "Test dataset and labels do not match."
        )

    # --------------------------------------------------------
    # Category columns
    # --------------------------------------------------------

    for category in CATEGORY_COLUMNS:

        if category not in train_df.columns:

            raise ValueError(
                f"Missing category column: {category}"
            )

    print(
        "Category labels validated successfully."
    )


# ============================================================
# CALCULATE CATEGORY RESULTS
# ============================================================

def calculate_category_results(
    y_true,
    y_pred
):

    results = []

    for index, category in enumerate(
        CATEGORY_COLUMNS
    ):

        true_values = y_true[:, index]

        predicted_values = y_pred[:, index]

        accuracy = accuracy_score(
            true_values,
            predicted_values
        )

        precision = precision_score(
            true_values,
            predicted_values,
            zero_division=0
        )

        recall = recall_score(
            true_values,
            predicted_values,
            zero_division=0
        )

        f1 = f1_score(
            true_values,
            predicted_values,
            zero_division=0
        )

        results.append({

            "category":
                category,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# PRINT CATEGORY RESULTS
# ============================================================

def print_category_results(
    results_df,
    title
):

    print(
        "\n" + "=" * 75
    )

    print(
        title
    )

    print(
        "=" * 75
    )

    for _, row in results_df.iterrows():

        print(
            f"\n{row['category']}"
        )

        print(
            f"  Accuracy  : "
            f"{row['accuracy']:.4f}"
        )

        print(
            f"  Precision : "
            f"{row['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{row['recall']:.4f}"
        )

        print(
            f"  F1 Score  : "
            f"{row['f1_score']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "STEP 9 — ROBUST MULTI-LABEL "
        "CATEGORY DETECTION"
    )

    print("=" * 75)

    # ========================================================
    # LOAD FEATURES
    # ========================================================

    (
        X_train,
        X_validation,
        X_test
    ) = load_features()

    # ========================================================
    # LOAD DATASETS
    # ========================================================

    (
        train_df,
        validation_df,
        test_df
    ) = load_datasets()

    print(
        "\nDataset splits loaded:"
    )

    print(
        f"Train rows      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows : "
        f"{len(validation_df)}"
    )

    print(
        f"Test rows       : "
        f"{len(test_df)}"
    )

    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    validate_features(
        X_train,
        X_validation,
        X_test
    )

    # ========================================================
    # PREPARE CATEGORY LABELS
    # ========================================================

    y_train = (
        train_df[
            CATEGORY_COLUMNS
        ]
        .astype(int)
        .values
    )

    y_validation = (
        validation_df[
            CATEGORY_COLUMNS
        ]
        .astype(int)
        .values
    )

    y_test = (
        test_df[
            CATEGORY_COLUMNS
        ]
        .astype(int)
        .values
    )

    validate_labels(
        train_df,
        validation_df,
        test_df,
        y_train,
        y_validation,
        y_test
    )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    print(
        "\nTraining category distribution:"
    )

    for index, category in enumerate(
        CATEGORY_COLUMNS
    ):

        positive_count = (
            y_train[:, index]
            .sum()
        )

        negative_count = (
            len(y_train)
            - positive_count
        )

        print(
            f"{category:20s} "
            f"positive={positive_count:4d} "
            f"negative={negative_count:4d}"
        )

    # ========================================================
    # TRAIN ONE-VS-REST LINEAR SVM
    # ========================================================

    print(
        "\n" + "-" * 75
    )

    print(
        "Training One-vs-Rest Linear SVM..."
    )

    print(
        "-" * 75
    )

    model = OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
        )
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # ========================================================
    # VALIDATION PREDICTION
    # ========================================================

    print(
        "\nRunning validation prediction..."
    )

    validation_pred = model.predict(
        X_validation
    )

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    print(
        "Running test prediction..."
    )

    test_pred = model.predict(
        X_test
    )

    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    validation_results_df = (
        calculate_category_results(
            y_validation,
            validation_pred
        )
    )

    print_category_results(
        validation_results_df,
        "VALIDATION RESULTS — CATEGORY WISE"
    )

    validation_results_df.to_csv(
        VALIDATION_REPORT_PATH,
        index=False
    )

    # ========================================================
    # TEST RESULTS
    # ========================================================

    test_results_df = (
        calculate_category_results(
            y_test,
            test_pred
        )
    )

    print_category_results(
        test_results_df,
        "TEST RESULTS — CATEGORY WISE"
    )

    test_results_df.to_csv(
        TEST_REPORT_PATH,
        index=False
    )

    # ========================================================
    # MULTI-LABEL TEST METRICS
    # ========================================================

    micro_f1 = f1_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    micro_precision = precision_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "MULTI-LABEL TEST SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"Micro Precision : "
        f"{micro_precision:.4f}"
    )

    print(
        f"Micro Recall    : "
        f"{micro_recall:.4f}"
    )

    print(
        f"Micro F1        : "
        f"{micro_f1:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1     : "
        f"{weighted_f1:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=CATEGORY_COLUMNS,
            zero_division=0
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_PATH
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = pd.DataFrame({

        "metric": [
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "weighted_f1",
        ],

        "value": [
            micro_precision,
            micro_recall,
            micro_f1,
            macro_f1,
            weighted_f1,
        ]
    })

    summary.to_csv(
        TEST_SUMMARY_PATH,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 75
    )

    print(
        f"Model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Validation report: "
        f"{VALIDATION_REPORT_PATH}"
    )

    print(
        f"Test report: "
        f"{TEST_REPORT_PATH}"
    )

    print(
        f"Test summary: "
        f"{TEST_SUMMARY_PATH}"
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 9 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()