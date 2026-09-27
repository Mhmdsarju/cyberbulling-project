import os
import joblib
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = (
    "data/processed/"
    "features_robust"
)

MODEL_DIR = "models"

REPORT_DIR = "reports"

MODEL_PATH = (
    "models/"
    "logistic_regression_robust_multilingual.joblib"
)

REPORT_PATH = (
    "reports/"
    "logistic_regression_robust_multilingual_validation.csv"
)

CONFUSION_MATRIX_PATH = (
    "reports/model_evaluation/"
    "logistic_regression_robust_multilingual_confusion_matrix.png"
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

os.makedirs(
    "reports/model_evaluation",
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

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training features not found: {train_path}"
        )

    if not os.path.exists(validation_path):
        raise FileNotFoundError(
            f"Validation features not found: {validation_path}"
        )

    X_train = load_npz(
        train_path
    )

    X_validation = load_npz(
        validation_path
    )

    return X_train, X_validation


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    train_label_path = os.path.join(
        FEATURE_DIR,
        "y_train_robust.joblib"
    )

    validation_label_path = os.path.join(
        FEATURE_DIR,
        "y_validation_robust.joblib"
    )

    if not os.path.exists(
        train_label_path
    ):
        raise FileNotFoundError(
            f"Training labels not found: "
            f"{train_label_path}"
        )

    if not os.path.exists(
        validation_label_path
    ):
        raise FileNotFoundError(
            f"Validation labels not found: "
            f"{validation_label_path}"
        )

    y_train = joblib.load(
        train_label_path
    )

    y_validation = joblib.load(
        validation_label_path
    )

    return y_train, y_validation


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 6 — ROBUST LOGISTIC REGRESSION")
    print("=" * 75)

    # ========================================================
    # LOAD FEATURES
    # ========================================================

    X_train, X_validation = load_features()

    # ========================================================
    # LOAD LABELS
    # ========================================================

    y_train, y_validation = load_labels()

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    print("\nFeature shapes:")

    print(
        f"Train      : "
        f"{X_train.shape}"
    )

    print(
        f"Validation : "
        f"{X_validation.shape}"
    )

    # ========================================================
    # FEATURE DIMENSION VALIDATION
    # ========================================================

    print("\nFeature dimension validation:")

    if X_train.shape[1] != X_validation.shape[1]:

        raise ValueError(
            "Train and validation feature dimensions "
            "do not match."
        )

    print(
        f"Feature dimension : "
        f"{X_train.shape[1]}"
    )

    print(
        "Train/Validation feature dimensions match."
    )

    # ========================================================
    # ROW COUNT VALIDATION
    # ========================================================

    if X_train.shape[0] != len(y_train):

        raise ValueError(
            "Training feature rows and labels "
            "do not match."
        )

    if X_validation.shape[0] != len(
        y_validation
    ):

        raise ValueError(
            "Validation feature rows and labels "
            "do not match."
        )

    print(
        "Feature/label row counts match."
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print("\nTraining distribution:")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation distribution:")

    print(
        pd.Series(y_validation)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\nCreating Logistic Regression model...")

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="liblinear",
        random_state=42,
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
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
        "\nGenerating validation predictions..."
    )

    y_pred = model.predict(
        X_validation
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_validation,
        y_pred
    )

    precision = precision_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_validation,
        y_pred
    )

    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("VALIDATION RESULTS")
    print("=" * 75)

    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\nConfusion Matrix:")

    print(cm)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_validation,
            y_pred,
            target_names=[
                "Safe",
                "Cyberbullying"
            ],
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

    print(
        "\nSaved model:"
    )

    print(
        MODEL_PATH
    )

    # ========================================================
    # SAVE VALIDATION REPORT
    # ========================================================

    results = pd.DataFrame({

        "model": [
            "Logistic Regression"
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
            accuracy
        ],

        "precision": [
            precision
        ],

        "recall": [
            recall
        ],

        "f1_score": [
            f1
        ],

        "true_negative": [
            cm[0, 0]
        ],

        "false_positive": [
            cm[0, 1]
        ],

        "false_negative": [
            cm[1, 0]
        ],

        "true_positive": [
            cm[1, 1]
        ],
    })

    results.to_csv(
        REPORT_PATH,
        index=False
    )

    print(
        "\nSaved validation report:"
    )

    print(
        REPORT_PATH
    )

    # ========================================================
    # SAVE CONFUSION MATRIX IMAGE
    # ========================================================

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Logistic Regression - Robust Validation"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        [
            "Safe",
            "Cyberbullying"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Safe",
            "Cyberbullying"
        ]
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150
    )

    plt.close()

    print(
        "\nSaved confusion matrix:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 6 COMPLETED SUCCESSFULLY"
    )
    print("=" * 75)

    print(
        f"Training samples      : "
        f"{X_train.shape[0]}"
    )

    print(
        f"Validation samples    : "
        f"{X_validation.shape[0]}"
    )

    print(
        f"Feature count         : "
        f"{X_train.shape[1]}"
    )

    print(
        f"Validation Accuracy   : "
        f"{accuracy:.4f}"
    )

    print(
        f"Validation Precision  : "
        f"{precision:.4f}"
    )

    print(
        f"Validation Recall     : "
        f"{recall:.4f}"
    )

    print(
        f"Validation F1 Score   : "
        f"{f1:.4f}"
    )

    print(
        "\nStep 6 completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()