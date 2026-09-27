import os
import joblib
import pandas as pd
import numpy as np

from scipy.sparse import load_npz

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits_realworld",
    "train.csv"
)

VALIDATION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits_realworld",
    "validation.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits_realworld",
    "test.csv"
)


# ============================================================
# FEATURE DIRECTORY
# ============================================================

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features_realworld"
)


# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "category_models"
)

INTENT_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "intent_models"
)

CONTENT_CATEGORY_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "content_category_models"
)


# ============================================================
# REPORT DIRECTORIES
# ============================================================

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "category_models"
)

INTENT_REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "intent_models"
)

CONTENT_CATEGORY_REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "content_category_models"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    INTENT_MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    CONTENT_CATEGORY_MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    INTENT_REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    CONTENT_CATEGORY_REPORT_DIR,
    exist_ok=True
)


# ============================================================
# EXISTING OFFENSE CATEGORIES
# ============================================================

OFFENSE_CATEGORY_COLUMNS = [

    "insult",

    "threat",

    "violence",

    "hate",

    "sexual_harassment",

    "sexual_abuse",

    "profanity",
]


# ============================================================
# INTENT CATEGORIES
# ============================================================

INTENT_LABELS = [

    "offensive",

    "informational",

    "requesting",

    "questioning",

    "praising",

    "defensive",

    "neutral",

    "supporting",

    "apologizing",

    "thanking",

    "begging",

    "casual",

    "greeting",
]


# ============================================================
# CONTENT CATEGORIES
# ============================================================

CONTENT_CATEGORY_LABELS = [

    "safe_communication",

    "none",

    "negative_media_feedback",

    "threat_violence",

    "sexual_abuse",

    "personal_attack",

    "positive_media_feedback",

    "bad_content",

    "hate_abuse",

    "profanity",

    "low_quality",

    "negative_review",

    "spam",

    "privacy",

    "misinformation",
]


# ============================================================
# MODEL PATHS
# ============================================================

OFFENSE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "category_multilabel_svm_realworld_multilingual.joblib"
)

INTENT_MODEL_PATH = os.path.join(
    INTENT_MODEL_DIR,
    "intent_svm_realworld_multilingual.joblib"
)

CONTENT_CATEGORY_MODEL_PATH = os.path.join(
    CONTENT_CATEGORY_MODEL_DIR,
    "content_category_svm_realworld_multilingual.joblib"
)


# ============================================================
# FEATURE PATHS
# ============================================================

TRAIN_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "X_train_tfidf_realworld.npz"
)

VALIDATION_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "X_validation_tfidf_realworld.npz"
)

TEST_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "X_test_tfidf_realworld.npz"
)


# ============================================================
# EXPECTED DATASET SIZE
# ============================================================

EXPECTED_TRAIN_ROWS = 20000

EXPECTED_VALIDATION_ROWS = 2500

EXPECTED_TEST_ROWS = 2500


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    print(
        "\nLoading TF-IDF features..."
    )

    for path in [
        TRAIN_FEATURE_PATH,
        VALIDATION_FEATURE_PATH,
        TEST_FEATURE_PATH,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Feature file not found: {path}"
            )

    X_train = load_npz(
        TRAIN_FEATURE_PATH
    )

    X_validation = load_npz(
        VALIDATION_FEATURE_PATH
    )

    X_test = load_npz(
        TEST_FEATURE_PATH
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

    return (
        X_train,
        X_validation,
        X_test,
    )


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print(
        "\nLoading datasets..."
    )

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

    print(
        f"Train rows      : {len(train_df)}"
    )

    print(
        f"Validation rows : {len(validation_df)}"
    )

    print(
        f"Test rows       : {len(test_df)}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# VALIDATE DATASET SIZE
# ============================================================

def validate_dataset_sizes(
    train_df,
    validation_df,
    test_df,
):

    expected = {

        "train": EXPECTED_TRAIN_ROWS,

        "validation": EXPECTED_VALIDATION_ROWS,

        "test": EXPECTED_TEST_ROWS,
    }

    actual = {

        "train": len(train_df),

        "validation": len(validation_df),

        "test": len(test_df),
    }

    for split_name in expected:

        if actual[split_name] != expected[split_name]:

            raise ValueError(
                f"{split_name} dataset has "
                f"{actual[split_name]} rows. "
                f"Expected {expected[split_name]}."
            )


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    X_train,
    X_validation,
    X_test,
):

    if X_train.shape[0] != EXPECTED_TRAIN_ROWS:

        raise ValueError(
            "Unexpected train feature row count."
        )

    if X_validation.shape[0] != EXPECTED_VALIDATION_ROWS:

        raise ValueError(
            "Unexpected validation feature row count."
        )

    if X_test.shape[0] != EXPECTED_TEST_ROWS:

        raise ValueError(
            "Unexpected test feature row count."
        )

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
        f"\nFeature dimension : {X_train.shape[1]}"
    )


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(
    train_df,
    validation_df,
    test_df,
):

    required_columns = (

        OFFENSE_CATEGORY_COLUMNS

        + [

            "intent",

            "content_category",

        ]
    )

    for column in required_columns:

        for split_name, df in [

            ("train", train_df),

            ("validation", validation_df),

            ("test", test_df),

        ]:

            if column not in df.columns:

                raise ValueError(
                    f"Missing column "
                    f"'{column}' "
                    f"in {split_name} dataset."
                )

    print(
        "\nRequired category columns validated."
    )


# ============================================================
# VALIDATE LABEL VALUES
# ============================================================

def validate_label_values(
    train_df,
    validation_df,
    test_df,
):

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    for split_name, df in [

        ("train", train_df),

        ("validation", validation_df),

        ("test", test_df),

    ]:

        found_intents = set(
            df["intent"]
            .dropna()
            .astype(str)
            .unique()
        )

        invalid_intents = (
            found_intents
            - set(INTENT_LABELS)
        )

        if invalid_intents:

            raise ValueError(
                f"Invalid intent labels "
                f"in {split_name}: "
                f"{invalid_intents}"
            )

    # --------------------------------------------------------
    # Content Category
    # --------------------------------------------------------

    for split_name, df in [

        ("train", train_df),

        ("validation", validation_df),

        ("test", test_df),

    ]:

        found_categories = set(
            df["content_category"]
            .dropna()
            .astype(str)
            .unique()
        )

        invalid_categories = (
            found_categories
            - set(CONTENT_CATEGORY_LABELS)
        )

        if invalid_categories:

            raise ValueError(
                f"Invalid content categories "
                f"in {split_name}: "
                f"{invalid_categories}"
            )

    # --------------------------------------------------------
    # Offense Categories
    # --------------------------------------------------------

    for split_name, df in [

        ("train", train_df),

        ("validation", validation_df),

        ("test", test_df),

    ]:

        for category in OFFENSE_CATEGORY_COLUMNS:

            values = set(
                df[category]
                .dropna()
                .astype(int)
                .unique()
            )

            if not values.issubset({0, 1}):

                raise ValueError(
                    f"Invalid values in "
                    f"{split_name}.{category}: "
                    f"{values}"
                )

    print(
        "All label values validated."
    )


# ============================================================
# TRAIN OFFENSE CATEGORY MODEL
# ============================================================

def train_offense_category_model(
    X_train,
    X_validation,
    X_test,
    train_df,
    validation_df,
    test_df,
):

    print(
        "\n" + "=" * 75
    )

    print(
        "1. TRAINING OFFENSE CATEGORY MODEL"
    )

    print(
        "=" * 75
    )

    y_train = (
        train_df[
            OFFENSE_CATEGORY_COLUMNS
        ]
        .astype(int)
        .values
    )

    y_validation = (
        validation_df[
            OFFENSE_CATEGORY_COLUMNS
        ]
        .astype(int)
        .values
    )

    y_test = (
        test_df[
            OFFENSE_CATEGORY_COLUMNS
        ]
        .astype(int)
        .values
    )

    print(
        "\nTraining category distribution:"
    )

    for index, category in enumerate(
        OFFENSE_CATEGORY_COLUMNS
    ):

        positive_count = int(
            y_train[:, index].sum()
        )

        print(
            f"{category:20s} "
            f"positive={positive_count}"
        )

    model = OneVsRestClassifier(

        LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
        )
    )

    print(
        "\nTraining One-vs-Rest Linear SVM..."
    )

    model.fit(
        X_train,
        y_train
    )

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    validation_rows = []

    test_rows = []

    for index, category in enumerate(
        OFFENSE_CATEGORY_COLUMNS
    ):

        validation_precision = precision_score(
            y_validation[:, index],
            validation_pred[:, index],
            zero_division=0,
        )

        validation_recall = recall_score(
            y_validation[:, index],
            validation_pred[:, index],
            zero_division=0,
        )

        validation_f1 = f1_score(
            y_validation[:, index],
            validation_pred[:, index],
            zero_division=0,
        )

        test_precision = precision_score(
            y_test[:, index],
            test_pred[:, index],
            zero_division=0,
        )

        test_recall = recall_score(
            y_test[:, index],
            test_pred[:, index],
            zero_division=0,
        )

        test_f1 = f1_score(
            y_test[:, index],
            test_pred[:, index],
            zero_division=0,
        )

        validation_rows.append({

            "category": category,

            "precision": validation_precision,

            "recall": validation_recall,

            "f1_score": validation_f1,
        })

        test_rows.append({

            "category": category,

            "precision": test_precision,

            "recall": test_recall,

            "f1_score": test_f1,
        })

    validation_results = pd.DataFrame(
        validation_rows
    )

    test_results = pd.DataFrame(
        test_rows
    )

    validation_results.to_csv(
        os.path.join(
            REPORT_DIR,
            "realworld_category_validation_results.csv"
        ),
        index=False
    )

    test_results.to_csv(
        os.path.join(
            REPORT_DIR,
            "realworld_category_test_results.csv"
        ),
        index=False
    )

    micro_precision = precision_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0,
    )

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

        ],
    })

    summary.to_csv(
        os.path.join(
            REPORT_DIR,
            "realworld_category_test_summary.csv"
        ),
        index=False
    )

    joblib.dump(
        model,
        OFFENSE_MODEL_PATH
    )

    print(
        "\nOffense category model saved:"
    )

    print(
        OFFENSE_MODEL_PATH
    )


# ============================================================
# TRAIN INTENT MODEL
# ============================================================

def train_intent_model(
    X_train,
    X_validation,
    X_test,
    train_df,
    validation_df,
    test_df,
):

    print(
        "\n" + "=" * 75
    )

    print(
        "2. TRAINING INTENT MODEL"
    )

    print(
        "=" * 75
    )

    y_train = (
        train_df["intent"]
        .astype(str)
        .values
    )

    y_validation = (
        validation_df["intent"]
        .astype(str)
        .values
    )

    y_test = (
        test_df["intent"]
        .astype(str)
        .values
    )

    print(
        "\nIntent distribution:"
    )

    print(
        train_df[
            "intent"
        ].value_counts()
    )

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42,
    )

    print(
        "\nTraining Intent Linear SVM..."
    )

    model.fit(
        X_train,
        y_train
    )

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    validation_report = classification_report(
        y_validation,
        validation_pred,
        labels=INTENT_LABELS,
        output_dict=True,
        zero_division=0,
    )

    test_report = classification_report(
        y_test,
        test_pred,
        labels=INTENT_LABELS,
        output_dict=True,
        zero_division=0,
    )

    validation_rows = []

    test_rows = []

    for label in INTENT_LABELS:

        validation_rows.append({

            "intent": label,

            "precision": validation_report[
                label
            ]["precision"],

            "recall": validation_report[
                label
            ]["recall"],

            "f1_score": validation_report[
                label
            ]["f1-score"],

            "support": validation_report[
                label
            ]["support"],
        })

        test_rows.append({

            "intent": label,

            "precision": test_report[
                label
            ]["precision"],

            "recall": test_report[
                label
            ]["recall"],

            "f1_score": test_report[
                label
            ]["f1-score"],

            "support": test_report[
                label
            ]["support"],
        })

    pd.DataFrame(
        validation_rows
    ).to_csv(
        os.path.join(
            INTENT_REPORT_DIR,
            "intent_realworld_validation_results.csv"
        ),
        index=False
    )

    pd.DataFrame(
        test_rows
    ).to_csv(
        os.path.join(
            INTENT_REPORT_DIR,
            "intent_realworld_test_results.csv"
        ),
        index=False
    )

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    summary = pd.DataFrame({

        "metric": [

            "accuracy",

            "macro_f1",

            "num_classes",

            "feature_dimension",

            "test_samples",
        ],

        "value": [

            test_accuracy,

            macro_f1,

            len(INTENT_LABELS),

            X_test.shape[1],

            len(y_test),
        ],
    })

    summary.to_csv(
        os.path.join(
            INTENT_REPORT_DIR,
            "intent_realworld_test_summary.csv"
        ),
        index=False
    )

    confusion = confusion_matrix(
        y_test,
        test_pred,
        labels=INTENT_LABELS,
    )

    pd.DataFrame(
        confusion,
        index=INTENT_LABELS,
        columns=INTENT_LABELS,
    ).to_csv(
        os.path.join(
            INTENT_REPORT_DIR,
            "intent_realworld_confusion_matrix.csv"
        )
    )

    joblib.dump(
        model,
        INTENT_MODEL_PATH
    )

    print(
        f"\nIntent validation accuracy : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Intent test accuracy       : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Intent test macro F1       : "
        f"{macro_f1:.4f}"
    )

    print(
        "\nIntent model saved:"
    )

    print(
        INTENT_MODEL_PATH
    )


# ============================================================
# TRAIN CONTENT CATEGORY MODEL
# ============================================================

def train_content_category_model(
    X_train,
    X_validation,
    X_test,
    train_df,
    validation_df,
    test_df,
):

    print(
        "\n" + "=" * 75
    )

    print(
        "3. TRAINING CONTENT CATEGORY MODEL"
    )

    print(
        "=" * 75
    )

    y_train = (
        train_df["content_category"]
        .astype(str)
        .values
    )

    y_validation = (
        validation_df["content_category"]
        .astype(str)
        .values
    )

    y_test = (
        test_df["content_category"]
        .astype(str)
        .values
    )

    print(
        "\nContent category distribution:"
    )

    print(
        train_df[
            "content_category"
        ].value_counts()
    )

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42,
    )

    print(
        "\nTraining Content Category Linear SVM..."
    )

    model.fit(
        X_train,
        y_train
    )

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    validation_report = classification_report(
        y_validation,
        validation_pred,
        labels=CONTENT_CATEGORY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    test_report = classification_report(
        y_test,
        test_pred,
        labels=CONTENT_CATEGORY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    validation_rows = []

    test_rows = []

    for label in CONTENT_CATEGORY_LABELS:

        validation_rows.append({

            "content_category": label,

            "precision": validation_report[
                label
            ]["precision"],

            "recall": validation_report[
                label
            ]["recall"],

            "f1_score": validation_report[
                label
            ]["f1-score"],

            "support": validation_report[
                label
            ]["support"],
        })

        test_rows.append({

            "content_category": label,

            "precision": test_report[
                label
            ]["precision"],

            "recall": test_report[
                label
            ]["recall"],

            "f1_score": test_report[
                label
            ]["f1-score"],

            "support": test_report[
                label
            ]["support"],
        })

    pd.DataFrame(
        validation_rows
    ).to_csv(
        os.path.join(
            CONTENT_CATEGORY_REPORT_DIR,
            "content_category_realworld_validation_results.csv"
        ),
        index=False
    )

    pd.DataFrame(
        test_rows
    ).to_csv(
        os.path.join(
            CONTENT_CATEGORY_REPORT_DIR,
            "content_category_realworld_test_results.csv"
        ),
        index=False
    )

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    summary = pd.DataFrame({

        "metric": [

            "accuracy",

            "macro_f1",

            "num_classes",

            "feature_dimension",

            "test_samples",
        ],

        "value": [

            test_accuracy,

            macro_f1,

            len(CONTENT_CATEGORY_LABELS),

            X_test.shape[1],

            len(y_test),
        ],
    })

    summary.to_csv(
        os.path.join(
            CONTENT_CATEGORY_REPORT_DIR,
            "content_category_realworld_test_summary.csv"
        ),
        index=False
    )

    confusion = confusion_matrix(
        y_test,
        test_pred,
        labels=CONTENT_CATEGORY_LABELS,
    )

    pd.DataFrame(
        confusion,
        index=CONTENT_CATEGORY_LABELS,
        columns=CONTENT_CATEGORY_LABELS,
    ).to_csv(
        os.path.join(
            CONTENT_CATEGORY_REPORT_DIR,
            "content_category_realworld_confusion_matrix.csv"
        )
    )

    joblib.dump(
        model,
        CONTENT_CATEGORY_MODEL_PATH
    )

    print(
        f"\nContent category validation accuracy : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Content category test accuracy       : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Content category test macro F1       : "
        f"{macro_f1:.4f}"
    )

    print(
        "\nContent category model saved:"
    )

    print(
        CONTENT_CATEGORY_MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "STEP 9 — COMPLETE CATEGORY CLASSIFICATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
    ) = load_features()

    (
        train_df,
        validation_df,
        test_df,
    ) = load_datasets()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_dataset_sizes(
        train_df,
        validation_df,
        test_df,
    )

    validate_features(
        X_train,
        X_validation,
        X_test,
    )

    validate_columns(
        train_df,
        validation_df,
        test_df,
    )

    validate_label_values(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------

    train_offense_category_model(
        X_train,
        X_validation,
        X_test,
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------

    train_intent_model(
        X_train,
        X_validation,
        X_test,
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # MODEL 3
    # --------------------------------------------------------

    train_content_category_model(
        X_train,
        X_validation,
        X_test,
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "ALL THREE CATEGORY MODELS TRAINED"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()