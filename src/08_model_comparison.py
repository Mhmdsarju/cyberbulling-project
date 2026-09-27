import os
import joblib
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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

REPORT_PATH = (
    "reports/"
    "robust_model_comparison_test.csv"
)

CONFUSION_MATRIX_DIR = (
    "reports/model_evaluation"
)


# ============================================================
# MODEL PATHS
# ============================================================

LOGISTIC_MODEL_PATH = (
    "models/"
    "logistic_regression_robust_multilingual.joblib"
)

SVM_MODEL_PATH = (
    "models/"
    "linear_svm_robust_multilingual.joblib"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    CONFUSION_MATRIX_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST FEATURES
# ============================================================

def load_test_features():

    test_features_path = os.path.join(
        FEATURE_DIR,
        "X_test_tfidf_robust.npz"
    )

    if not os.path.exists(
        test_features_path
    ):
        raise FileNotFoundError(
            f"Test features not found: "
            f"{test_features_path}"
        )

    return load_npz(
        test_features_path
    )


# ============================================================
# LOAD TEST LABELS
# ============================================================

def load_test_labels():

    test_label_path = os.path.join(
        FEATURE_DIR,
        "y_test_robust.joblib"
    )

    if not os.path.exists(
        test_label_path
    ):
        raise FileNotFoundError(
            f"Test labels not found: "
            f"{test_label_path}"
        )

    return joblib.load(
        test_label_path
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not os.path.exists(
        LOGISTIC_MODEL_PATH
    ):
        raise FileNotFoundError(
            "Logistic Regression model not found: "
            f"{LOGISTIC_MODEL_PATH}"
        )

    if not os.path.exists(
        SVM_MODEL_PATH
    ):
        raise FileNotFoundError(
            "Linear SVM model not found: "
            f"{SVM_MODEL_PATH}"
        )

    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )

    svm_model = joblib.load(
        SVM_MODEL_PATH
    )

    return {
        "Logistic Regression":
            logistic_model,

        "Linear SVM":
            svm_model,
    }


# ============================================================
# VALIDATE TEST DATA
# ============================================================

def validate_test_data(
    X_test,
    y_test
):

    print(
        "\nTest feature validation:"
    )

    print(
        f"Feature rows : "
        f"{X_test.shape[0]}"
    )

    print(
        f"Feature cols : "
        f"{X_test.shape[1]}"
    )

    print(
        f"Label rows   : "
        f"{len(y_test)}"
    )

    if X_test.shape[0] != len(
        y_test
    ):
        raise ValueError(
            "Test feature rows and test "
            "labels do not match."
        )

    print(
        "Test feature/label row counts match."
    )


# ============================================================
# VALIDATE MODEL FEATURES
# ============================================================

def validate_model_features(
    models,
    X_test
):

    print(
        "\nModel feature validation:"
    )

    for model_name, model in models.items():

        if not hasattr(
            model,
            "n_features_in_"
        ):
            raise ValueError(
                f"{model_name} does not expose "
                "n_features_in_."
            )

        model_features = (
            model.n_features_in_
        )

        print(
            f"{model_name}: "
            f"{model_features} features"
        )

        if model_features != X_test.shape[1]:

            raise ValueError(
                f"{model_name} expects "
                f"{model_features} features, "
                f"but test data has "
                f"{X_test.shape[1]} features."
            )

    print(
        "All model/test feature dimensions match."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 8 — ROBUST MODEL COMPARISON")
    print("=" * 75)

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    X_test = load_test_features()

    y_test = load_test_labels()

    print(
        "\nTest dataset:"
    )

    print(
        f"Samples  : "
        f"{len(y_test)}"
    )

    print(
        f"Features : "
        f"{X_test.shape}"
    )

    # ========================================================
    # VALIDATE TEST DATA
    # ========================================================

    validate_test_data(
        X_test,
        y_test
    )

    # ========================================================
    # TEST DISTRIBUTION
    # ========================================================

    print(
        "\nTest distribution:"
    )

    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    models = load_models()

    print(
        "\nModels loaded successfully:"
    )

    for model_name in models:

        print(
            f"- {model_name}"
        )

    # ========================================================
    # VALIDATE MODEL FEATURE DIMENSIONS
    # ========================================================

    validate_model_features(
        models,
        X_test
    )

    # ========================================================
    # EVALUATE MODELS
    # ========================================================

    results = []

    for model_name, model in models.items():

        print(
            "\n" + "-" * 75
        )

        print(
            model_name
        )

        print(
            "-" * 75
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

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

        print(
            "\nConfusion Matrix:"
        )

        print(
            cm
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            "model":
                model_name,

            "dataset":
                "robust_test",

            "samples":
                len(y_test),

            "features":
                X_test.shape[1],

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "true_negative":
                cm[0, 0],

            "false_positive":
                cm[0, 1],

            "false_negative":
                cm[1, 0],

            "true_positive":
                cm[1, 1],
        })

        # ====================================================
        # SAVE CONFUSION MATRIX
        # ====================================================

        import matplotlib.pyplot as plt

        safe_name = (
            model_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        confusion_path = os.path.join(
            CONFUSION_MATRIX_DIR,
            f"{safe_name}_robust_test_confusion_matrix.png"
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            cm
        )

        plt.title(
            f"{model_name} - Robust Test"
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
            confusion_path,
            dpi=150
        )

        plt.close()

        print(
            "\nSaved confusion matrix:"
        )

        print(
            confusion_path
        )

    # ========================================================
    # CREATE COMPARISON DATAFRAME
    # ========================================================

    comparison_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SAVE COMPARISON REPORT
    # ========================================================

    comparison_df.to_csv(
        REPORT_PATH,
        index=False
    )

    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 75
    )

    print(
        comparison_df[
            [
                "model",
                "samples",
                "features",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\nSaved comparison report:"
    )

    print(
        REPORT_PATH
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 8 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()