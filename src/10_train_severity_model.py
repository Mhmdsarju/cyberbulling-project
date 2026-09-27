import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
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

MODEL_DIR = "models"

REPORT_DIR = (
    "reports/severity"
)

CHART_DIR = (
    "reports/severity"
)

MODEL_PATH = (
    f"{MODEL_DIR}/"
    "severity_svm_robust_multilingual.joblib"
)

VALIDATION_REPORT_PATH = (
    f"{REPORT_DIR}/"
    "severity_robust_validation_results.csv"
)

TEST_REPORT_PATH = (
    f"{REPORT_DIR}/"
    "severity_robust_test_results.csv"
)

CONFUSION_MATRIX_PATH = (
    f"{CHART_DIR}/"
    "severity_robust_confusion_matrix.png"
)

RANDOM_STATE = 42

SEVERITY_LABELS = [
    "none",
    "low",
    "medium",
    "high",
]


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

os.makedirs(
    CHART_DIR,
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

    if X_train.shape[0] != 7220:

        raise ValueError(
            "Unexpected training feature row count."
        )

    if X_validation.shape[0] != 902:

        raise ValueError(
            "Unexpected validation feature row count."
        )

    if X_test.shape[0] != 903:

        raise ValueError(
            "Unexpected test feature row count."
        )

    print(
        "Feature dimensions and row counts are valid."
    )


# ============================================================
# VALIDATE SEVERITY LABELS
# ============================================================

def validate_severity_labels(
    train_df,
    validation_df,
    test_df
):

    for df_name, df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        if "severity" not in df.columns:

            raise ValueError(
                f"Severity column missing in {df_name} dataset."
            )

        invalid_values = set(
            df["severity"].dropna().unique()
        ) - set(
            SEVERITY_LABELS
        )

        if invalid_values:

            raise ValueError(
                f"Invalid severity values in "
                f"{df_name}: {invalid_values}"
            )

    print(
        "Severity labels validated successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "STEP 10 — ROBUST MULTILINGUAL "
        "SEVERITY DETECTION"
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
    # FEATURE VALIDATION
    # ========================================================

    validate_features(
        X_train,
        X_validation,
        X_test
    )

    # ========================================================
    # SEVERITY LABEL VALIDATION
    # ========================================================

    validate_severity_labels(
        train_df,
        validation_df,
        test_df
    )

    # ========================================================
    # PREPARE LABELS
    # ========================================================

    y_train = (
        train_df["severity"]
        .astype(str)
    )

    y_validation = (
        validation_df["severity"]
        .astype(str)
    )

    y_test = (
        test_df["severity"]
        .astype(str)
    )

    # ========================================================
    # SEVERITY DISTRIBUTION
    # ========================================================

    print(
        "\nTraining severity distribution:"
    )

    print(
        y_train.value_counts()
        .reindex(
            SEVERITY_LABELS,
            fill_value=0
        )
        .to_string()
    )

    print(
        "\nValidation severity distribution:"
    )

    print(
        y_validation.value_counts()
        .reindex(
            SEVERITY_LABELS,
            fill_value=0
        )
        .to_string()
    )

    print(
        "\nTest severity distribution:"
    )

    print(
        y_test.value_counts()
        .reindex(
            SEVERITY_LABELS,
            fill_value=0
        )
        .to_string()
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print(
        "\nCreating Severity Linear SVM model..."
    )

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining Severity Linear SVM..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\nRunning validation prediction..."
    )

    validation_pred = model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    validation_precision = precision_score(
        y_validation,
        validation_pred,
        labels=SEVERITY_LABELS,
        average="macro",
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_pred,
        labels=SEVERITY_LABELS,
        average="macro",
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_pred,
        labels=SEVERITY_LABELS,
        average="macro",
        zero_division=0
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"Accuracy       : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{validation_precision:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{validation_recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{validation_f1:.4f}"
    )

    print(
        "\nValidation Classification Report:"
    )

    print(
        classification_report(
            y_validation,
            validation_pred,
            labels=SEVERITY_LABELS,
            target_names=SEVERITY_LABELS,
            zero_division=0
        )
    )

    # ========================================================
    # SAVE VALIDATION REPORT
    # ========================================================

    validation_metrics = pd.DataFrame({

        "model": [
            "Linear SVM"
        ],

        "dataset": [
            "robust_validation"
        ],

        "samples": [
            len(y_validation)
        ],

        "features": [
            X_train.shape[1]
        ],

        "accuracy": [
            validation_accuracy
        ],

        "macro_precision": [
            validation_precision
        ],

        "macro_recall": [
            validation_recall
        ],

        "macro_f1": [
            validation_f1
        ],
    })

    validation_metrics.to_csv(
        VALIDATION_REPORT_PATH,
        index=False
    )

    # ========================================================
    # TEST
    # ========================================================

    print(
        "\nRunning final test prediction..."
    )

    test_pred = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_precision = precision_score(
        y_test,
        test_pred,
        labels=SEVERITY_LABELS,
        average="macro",
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        labels=SEVERITY_LABELS,
        average="macro",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        labels=SEVERITY_LABELS,
        average="macro",
        zero_division=0
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"Accuracy       : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{test_recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{test_f1:.4f}"
    )

    print(
        "\nTest Classification Report:"
    )

    print(
        classification_report(
            y_test,
            test_pred,
            labels=SEVERITY_LABELS,
            target_names=SEVERITY_LABELS,
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=SEVERITY_LABELS
    )

    print(
        "\nTest Confusion Matrix:"
    )

    print(
        cm
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=SEVERITY_LABELS
    )

    display.plot()

    plt.title(
        "Robust Multilingual Severity Detection "
        "- Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150
    )

    plt.close()

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_PATH
    )

    # ========================================================
    # SAVE TEST METRICS
    # ========================================================

    test_metrics = pd.DataFrame({

        "model": [
            "Linear SVM"
        ],

        "dataset": [
            "robust_test"
        ],

        "samples": [
            len(y_test)
        ],

        "features": [
            X_train.shape[1]
        ],

        "accuracy": [
            test_accuracy
        ],

        "macro_precision": [
            test_precision
        ],

        "macro_recall": [
            test_recall
        ],

        "macro_f1": [
            test_f1
        ],
    })

    test_metrics.to_csv(
        TEST_REPORT_PATH,
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
        f"Model  : "
        f"{MODEL_PATH}"
    )

    print(
        f"Validation Metrics: "
        f"{VALIDATION_REPORT_PATH}"
    )

    print(
        f"Test Metrics: "
        f"{TEST_REPORT_PATH}"
    )

    print(
        f"Chart  : "
        f"{CONFUSION_MATRIX_PATH}"
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 10 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()