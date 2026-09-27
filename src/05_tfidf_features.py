import os
import re
import joblib
import pandas as pd

from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = (
    "data/processed/"
    "splits_realworld/train.csv"
)

VALIDATION_PATH = (
    "data/processed/"
    "splits_realworld/validation.csv"
)

TEST_PATH = (
    "data/processed/"
    "splits_realworld/test.csv"
)

FEATURE_DIR = (
    "data/processed/"
    "features_realworld"
)

MODEL_DIR = "models"


# ============================================================
# TF-IDF CONFIG
# ============================================================

# ------------------------------------------------------------
# Word-level features
# ------------------------------------------------------------

WORD_MAX_FEATURES = 10000
WORD_NGRAM_RANGE = (1, 2)
WORD_MIN_DF = 2


# ------------------------------------------------------------
# Character-level features
# ------------------------------------------------------------

CHAR_MAX_FEATURES = 15000
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2


# ------------------------------------------------------------
# No-space character-level features
# ------------------------------------------------------------

NOSPACE_CHAR_MAX_FEATURES = 15000
NOSPACE_CHAR_NGRAM_RANGE = (3, 5)
NOSPACE_CHAR_MIN_DF = 2


SUBLINEAR_TF = True


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    FEATURE_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(
        path,
        encoding="utf-8-sig"
    )


# ============================================================
# NO-SPACE TEXT PREPROCESSING
# ============================================================

def create_no_space_text(text):

    """
    Creates an additional text representation.

    Examples:

        "i love you"
        -> "iloveyou"

        "i-love-you"
        -> "iloveyou"

        "nee romba loosu da"
        -> "neerombaloosuda"

    Original text is not modified.
    """

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        "",
        text
    )

    text = re.sub(
        r"[-_]+",
        "",
        text
    )

    return text


def create_no_space_series(series):

    return series.apply(
        create_no_space_text
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 5 — MULTI-CHANNEL TF-IDF FEATURES")
    print("=" * 75)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_dataset(
        TRAIN_PATH
    )

    validation_df = load_dataset(
        VALIDATION_PATH
    )

    test_df = load_dataset(
        TEST_PATH
    )

    print("\nDataset sizes:")

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(validation_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Validate text column
    # --------------------------------------------------------

    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:

        if "text" not in df.columns:
            raise ValueError(
                f"'text' column missing in {name} dataset."
            )

    # ========================================================
    # ORIGINAL TEXT
    # ========================================================

    X_train_text = (
        train_df["text"]
        .fillna("")
        .astype(str)
    )

    X_validation_text = (
        validation_df["text"]
        .fillna("")
        .astype(str)
    )

    X_test_text = (
        test_df["text"]
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # NO-SPACE TEXT
    # ========================================================

    print("\n" + "-" * 75)
    print("CREATING NO-SPACE TEXT REPRESENTATION")
    print("-" * 75)

    X_train_nospace = create_no_space_series(
        X_train_text
    )

    X_validation_nospace = create_no_space_series(
        X_validation_text
    )

    X_test_nospace = create_no_space_series(
        X_test_text
    )

    print("\nExamples:")

    for i in range(
        min(5, len(X_train_text))
    ):

        print(
            f"Original : {X_train_text.iloc[i]}"
        )

        print(
            f"No-space : {X_train_nospace.iloc[i]}"
        )

        print()

    # ========================================================
    # CHANNEL 1 — WORD TF-IDF
    # ========================================================

    print("-" * 75)
    print("CHANNEL 1 — WORD TF-IDF")
    print("-" * 75)

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=WORD_NGRAM_RANGE,
        min_df=WORD_MIN_DF,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
    )

    # Fit ONLY on training data
    X_train_word = (
        word_vectorizer.fit_transform(
            X_train_text
        )
    )

    X_validation_word = (
        word_vectorizer.transform(
            X_validation_text
        )
    )

    X_test_word = (
        word_vectorizer.transform(
            X_test_text
        )
    )

    print(
        f"Word vocabulary : "
        f"{len(word_vectorizer.vocabulary_)}"
    )

    print(
        f"Train           : "
        f"{X_train_word.shape}"
    )

    print(
        f"Validation      : "
        f"{X_validation_word.shape}"
    )

    print(
        f"Test            : "
        f"{X_test_word.shape}"
    )

    # ========================================================
    # CHANNEL 2 — NORMAL CHARACTER TF-IDF
    # ========================================================

    print("\n" + "-" * 75)
    print("CHANNEL 2 — NORMAL CHARACTER TF-IDF")
    print("-" * 75)

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=CHAR_MIN_DF,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
    )

    # Fit ONLY on training data
    X_train_char = (
        char_vectorizer.fit_transform(
            X_train_text
        )
    )

    X_validation_char = (
        char_vectorizer.transform(
            X_validation_text
        )
    )

    X_test_char = (
        char_vectorizer.transform(
            X_test_text
        )
    )

    print(
        f"Character vocabulary : "
        f"{len(char_vectorizer.vocabulary_)}"
    )

    print(
        f"Train                 : "
        f"{X_train_char.shape}"
    )

    print(
        f"Validation            : "
        f"{X_validation_char.shape}"
    )

    print(
        f"Test                  : "
        f"{X_test_char.shape}"
    )

    # ========================================================
    # CHANNEL 3 — NO-SPACE CHARACTER TF-IDF
    # ========================================================

    print("\n" + "-" * 75)
    print("CHANNEL 3 — NO-SPACE CHARACTER TF-IDF")
    print("-" * 75)

    nospace_char_vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=NOSPACE_CHAR_NGRAM_RANGE,
        min_df=NOSPACE_CHAR_MIN_DF,
        max_features=NOSPACE_CHAR_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
    )

    # Fit ONLY on training data
    X_train_nospace_char = (
        nospace_char_vectorizer.fit_transform(
            X_train_nospace
        )
    )

    X_validation_nospace_char = (
        nospace_char_vectorizer.transform(
            X_validation_nospace
        )
    )

    X_test_nospace_char = (
        nospace_char_vectorizer.transform(
            X_test_nospace
        )
    )

    print(
        f"No-space character vocabulary : "
        f"{len(nospace_char_vectorizer.vocabulary_)}"
    )

    print(
        f"Train                         : "
        f"{X_train_nospace_char.shape}"
    )

    print(
        f"Validation                    : "
        f"{X_validation_nospace_char.shape}"
    )

    print(
        f"Test                          : "
        f"{X_test_nospace_char.shape}"
    )

    # ========================================================
    # COMBINE ALL THREE CHANNELS
    # ========================================================

    print("\n" + "-" * 75)
    print("COMBINING ALL THREE TF-IDF CHANNELS")
    print("-" * 75)

    X_train = hstack(
        [
            X_train_word,
            X_train_char,
            X_train_nospace_char,
        ]
    ).tocsr()

    X_validation = hstack(
        [
            X_validation_word,
            X_validation_char,
            X_validation_nospace_char,
        ]
    ).tocsr()

    X_test = hstack(
        [
            X_test_word,
            X_test_char,
            X_test_nospace_char,
        ]
    ).tocsr()

    print(
        f"Combined train features      : "
        f"{X_train.shape}"
    )

    print(
        f"Combined validation features : "
        f"{X_validation.shape}"
    )

    print(
        f"Combined test features       : "
        f"{X_test.shape}"
    )

    # ========================================================
    # SAVE VECTORIZERS
    # ========================================================

    word_vectorizer_path = os.path.join(
        MODEL_DIR,
        "tfidf_word_realworld_vectorizer.joblib"
    )

    char_vectorizer_path = os.path.join(
        MODEL_DIR,
        "tfidf_char_realworld_vectorizer.joblib"
    )

    nospace_char_vectorizer_path = os.path.join(
        MODEL_DIR,
        "tfidf_nospace_char_realworld_vectorizer.joblib"
    )

    joblib.dump(
        word_vectorizer,
        word_vectorizer_path
    )

    joblib.dump(
        char_vectorizer,
        char_vectorizer_path
    )

    joblib.dump(
        nospace_char_vectorizer,
        nospace_char_vectorizer_path
    )

    # ========================================================
    # SAVE COMBINED FEATURES
    # ========================================================

    train_features_path = os.path.join(
        FEATURE_DIR,
        "X_train_tfidf_realworld.npz"
    )

    validation_features_path = os.path.join(
        FEATURE_DIR,
        "X_validation_tfidf_realworld.npz"
    )

    test_features_path = os.path.join(
        FEATURE_DIR,
        "X_test_tfidf_realworld.npz"
    )

    save_npz(
        train_features_path,
        X_train
    )

    save_npz(
        validation_features_path,
        X_validation
    )

    save_npz(
        test_features_path,
        X_test
    )

    # ========================================================
    # SAVE LABELS
    # ========================================================

    y_train = (
        train_df["cyberbullying"]
        .astype(int)
    )

    y_validation = (
        validation_df["cyberbullying"]
        .astype(int)
    )

    y_test = (
        test_df["cyberbullying"]
        .astype(int)
    )

    joblib.dump(
        y_train,
        os.path.join(
            FEATURE_DIR,
            "y_train_realworld.joblib"
        )
    )

    joblib.dump(
        y_validation,
        os.path.join(
            FEATURE_DIR,
            "y_validation_realworld.joblib"
        )
    )

    joblib.dump(
        y_test,
        os.path.join(
            FEATURE_DIR,
            "y_test_realworld.joblib"
        )
    )

    # ========================================================
    # SAVE FEATURE INFORMATION
    # ========================================================

    feature_info = {

        "word_vocabulary_size":
            len(
                word_vectorizer.vocabulary_
            ),

        "character_vocabulary_size":
            len(
                char_vectorizer.vocabulary_
            ),

        "nospace_character_vocabulary_size":
            len(
                nospace_char_vectorizer.vocabulary_
            ),

        "word_ngram_range":
            WORD_NGRAM_RANGE,

        "character_ngram_range":
            CHAR_NGRAM_RANGE,

        "nospace_character_ngram_range":
            NOSPACE_CHAR_NGRAM_RANGE,

        "word_min_df":
            WORD_MIN_DF,

        "character_min_df":
            CHAR_MIN_DF,

        "nospace_character_min_df":
            NOSPACE_CHAR_MIN_DF,

        "word_max_features":
            WORD_MAX_FEATURES,

        "character_max_features":
            CHAR_MAX_FEATURES,

        "nospace_character_max_features":
            NOSPACE_CHAR_MAX_FEATURES,

        "combined_feature_count":
            X_train.shape[1],

        "train_rows":
            X_train.shape[0],

        "validation_rows":
            X_validation.shape[0],

        "test_rows":
            X_test.shape[0],
    }

    joblib.dump(
        feature_info,
        os.path.join(
            FEATURE_DIR,
            "feature_info_realworld.joblib"
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("TF-IDF FEATURE SUMMARY")
    print("=" * 75)

    print(
        f"Word vocabulary               : "
        f"{len(word_vectorizer.vocabulary_)}"
    )

    print(
        f"Character vocabulary          : "
        f"{len(char_vectorizer.vocabulary_)}"
    )

    print(
        f"No-space character vocabulary : "
        f"{len(nospace_char_vectorizer.vocabulary_)}"
    )

    print(
        f"Combined features             : "
        f"{X_train.shape[1]}"
    )

    print(
        f"Train shape                   : "
        f"{X_train.shape}"
    )

    print(
        f"Validation shape              : "
        f"{X_validation.shape}"
    )

    print(
        f"Test shape                    : "
        f"{X_test.shape}"
    )

    print("\nSaved vectorizers:")

    print(word_vectorizer_path)
    print(char_vectorizer_path)
    print(nospace_char_vectorizer_path)

    print("\nSaved features:")

    print(train_features_path)
    print(validation_features_path)
    print(test_features_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()