import os
import sys
import joblib
import importlib.util

from scipy.sparse import hstack


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(
        0,
        BASE_DIR
    )


# ============================================================
# LOAD EXPLANATION ENGINE
# ============================================================

EXPLANATION_ENGINE_PATH = os.path.join(
    BASE_DIR,
    "src",
    "12_explanation_engine.py"
)

spec = importlib.util.spec_from_file_location(
    "explanation_engine",
    EXPLANATION_ENGINE_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        "Could not load explanation engine."
    )

explanation_engine = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    explanation_engine
)

build_prediction_result = (
    explanation_engine.build_prediction_result
)


# ============================================================
# MODEL PATHS
# ============================================================

WORD_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_word_realworld_vectorizer.joblib"
)

CHAR_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_char_realworld_vectorizer.joblib"
)

NOSPACE_CHAR_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_nospace_char_realworld_vectorizer.joblib"
)


# ============================================================
# CYBERBULLYING MODEL
# ============================================================

CYBERBULLYING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_realworld_multilingual.joblib"
)


# ============================================================
# OFFENSE CATEGORY MODEL
# ============================================================

CATEGORY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "category_models",
    "category_multilabel_svm_realworld_multilingual.joblib"
)


# ============================================================
# INTENT MODEL
# ============================================================

INTENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "intent_models",
    "intent_svm_realworld_multilingual.joblib"
)


# ============================================================
# CONTENT CATEGORY MODEL
# ============================================================

CONTENT_CATEGORY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "content_category_models",
    "content_category_svm_realworld_multilingual.joblib"
)


# ============================================================
# SEVERITY MODEL
# ============================================================

SEVERITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "severity_svm_realworld_multilingual.joblib"
)


# ============================================================
# TARGET TYPE MODEL
# ============================================================

TARGET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "target_type_svm_realworld_multilingual.joblib"
)


# ============================================================
# EXPECTED FEATURE DIMENSION
# ============================================================

EXPECTED_FEATURE_DIMENSION = 27665


# ============================================================
# OFFENSE CATEGORY LABELS
# ============================================================

CATEGORY_LABELS = [

    "insult",

    "threat",

    "violence",

    "hate",

    "sexual_harassment",

    "sexual_abuse",

    "profanity",
]


# ============================================================
# INTENT LABELS
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
# CONTENT CATEGORY LABELS
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
# NO-SPACE TEXT TRANSFORMATION
# ============================================================

def create_no_space_text(text):

    """
    Create the same no-space representation
    used during Step 5 TF-IDF feature generation.

    Example:

        i love you
        ->
        iloveyou

        i-love-you
        ->
        iloveyou
    """

    text = str(
        text
    ).lower()

    text = "".join(
        text.split()
    )

    text = text.replace(
        "-",
        ""
    )

    text = text.replace(
        "_",
        ""
    )

    return text


# ============================================================
# PREDICTION PIPELINE
# ============================================================

class PredictionPipeline:

    def __init__(self):

        # ====================================================
        # WORD TF-IDF
        # ====================================================

        self.word_vectorizer = joblib.load(
            WORD_VECTORIZER_PATH
        )

        # ====================================================
        # CHARACTER TF-IDF
        # ====================================================

        self.char_vectorizer = joblib.load(
            CHAR_VECTORIZER_PATH
        )

        # ====================================================
        # NO-SPACE CHARACTER TF-IDF
        # ====================================================

        self.nospace_char_vectorizer = joblib.load(
            NOSPACE_CHAR_VECTORIZER_PATH
        )

        # ====================================================
        # CYBERBULLYING MODEL
        # ====================================================

        self.cyberbullying_model = joblib.load(
            CYBERBULLYING_MODEL_PATH
        )

        # ====================================================
        # OFFENSE CATEGORY MODEL
        # ====================================================

        self.category_model = joblib.load(
            CATEGORY_MODEL_PATH
        )

        # ====================================================
        # INTENT MODEL
        # ====================================================

        self.intent_model = joblib.load(
            INTENT_MODEL_PATH
        )

        # ====================================================
        # CONTENT CATEGORY MODEL
        # ====================================================

        self.content_category_model = joblib.load(
            CONTENT_CATEGORY_MODEL_PATH
        )

        # ====================================================
        # SEVERITY MODEL
        # ====================================================

        self.severity_model = joblib.load(
            SEVERITY_MODEL_PATH
        )

        # ====================================================
        # TARGET MODEL
        # ====================================================

        self.target_model = joblib.load(
            TARGET_MODEL_PATH
        )

        # ====================================================
        # VALIDATE MODEL FEATURE DIMENSIONS
        # ====================================================

        model_dimensions = {

            "Cyberbullying":
                self.cyberbullying_model
                .n_features_in_,

            "Category":
                self.category_model
                .estimators_[0]
                .n_features_in_,

            "Intent":
                self.intent_model
                .n_features_in_,

            "Content Category":
                self.content_category_model
                .n_features_in_,

            "Severity":
                self.severity_model
                .n_features_in_,

            "Target":
                self.target_model
                .n_features_in_,
        }

        print(
            "Model feature dimensions:"
        )

        for name, dimension in (
            model_dimensions.items()
        ):

            print(
                f"{name:20s}: {dimension}"
            )

            if (
                dimension
                != EXPECTED_FEATURE_DIMENSION
            ):

                raise ValueError(
                    f"{name} model expects "
                    f"{dimension} features, but "
                    f"pipeline expects "
                    f"{EXPECTED_FEATURE_DIMENSION}."
                )


    # ========================================================
    # CREATE FEATURES
    # ========================================================

    def create_features(self, text):

        texts = [
            text
        ]

        no_space_text = (
            create_no_space_text(
                text
            )
        )

        no_space_texts = [
            no_space_text
        ]

        # ----------------------------------------------------
        # Word TF-IDF
        # ----------------------------------------------------

        word_features = (
            self.word_vectorizer.transform(
                texts
            )
        )

        # ----------------------------------------------------
        # Character TF-IDF
        # ----------------------------------------------------

        char_features = (
            self.char_vectorizer.transform(
                texts
            )
        )

        # ----------------------------------------------------
        # No-space Character TF-IDF
        # ----------------------------------------------------

        no_space_char_features = (
            self.nospace_char_vectorizer.transform(
                no_space_texts
            )
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        combined_features = hstack(
            [
                word_features,
                char_features,
                no_space_char_features,
            ],
            format="csr"
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if (
            combined_features.shape[1]
            != EXPECTED_FEATURE_DIMENSION
        ):

            raise ValueError(
                "Generated feature dimension "
                f"{combined_features.shape[1]} "
                "does not match expected "
                f"{EXPECTED_FEATURE_DIMENSION}."
            )

        return combined_features


    # ========================================================
    # CONFIDENCE
    # ========================================================

    def calculate_confidence(
        self,
        features,
        cyber_prediction
    ):

        """
        Logistic Regression probability.

        This represents the probability assigned
        by the model to the predicted class.

        It is not a calibrated confidence score.
        """

        probabilities = (
            self.cyberbullying_model
            .predict_proba(
                features
            )
        )

        if cyber_prediction:

            confidence = probabilities[0][1]

        else:

            confidence = probabilities[0][0]

        return float(
            round(
                confidence,
                4
            )
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, text):

        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        if not isinstance(
            text,
            str
        ):

            raise TypeError(
                "Input text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Input text cannot be empty."
            )

        # ====================================================
        # FEATURE CREATION
        # ====================================================

        features = self.create_features(
            text
        )

        # ====================================================
        # 1. CYBERBULLYING
        # ====================================================

        cyber_prediction = (

            self.cyberbullying_model
            .predict(
                features
            )[0]
        )

        cyber_prediction = bool(
            cyber_prediction
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = (
            self.calculate_confidence(
                features,
                cyber_prediction
            )
        )

        # ====================================================
        # 2. OFFENSE CATEGORY
        # ====================================================

        category_prediction = (

            self.category_model
            .predict(
                features
            )[0]
        )

        categories = [

            CATEGORY_LABELS[index]

            for index, value
            in enumerate(
                category_prediction
            )

            if value == 1
        ]

        # ====================================================
        # 3. INTENT
        # ====================================================

        intent = (

            self.intent_model
            .predict(
                features
            )[0]
        )

        intent = str(
            intent
        )

        # ====================================================
        # 4. CONTENT CATEGORY
        # ====================================================

        content_category = (

            self.content_category_model
            .predict(
                features
            )[0]
        )

        content_category = str(
            content_category
        )

        # ====================================================
        # 5. SEVERITY
        # ====================================================

        severity = (

            self.severity_model
            .predict(
                features
            )[0]
        )

        severity = str(
            severity
        )

        # ====================================================
        # 6. TARGET TYPE
        # ====================================================

        target_type = (

            self.target_model
            .predict(
                features
            )[0]
        )

        target_type = str(
            target_type
        )

        # ====================================================
        # SAFE RESULT NORMALIZATION
        # ====================================================

        if not cyber_prediction:

            categories = []

            severity = "none"

            target_type = "none"

        # ====================================================
        # 7. EXPLANATION ENGINE
        # ====================================================

        result = build_prediction_result(

            text=text,

            cyberbullying=cyber_prediction,

            categories=categories,

            severity=severity,

            target_type=target_type,

            confidence=confidence,
        )

        # ====================================================
        # ADD INTENT + CONTENT CATEGORY
        # ====================================================

        result["intent"] = intent

        result["content_category"] = (
            content_category
        )

        return result


# ============================================================
# TEST
# ============================================================

def main():

    pipeline = PredictionPipeline()

    test_messages = [

        "nee romba loosu da",

        "unakku violence panniduven",

        "Ithu nalla useful information da",

        "iloveyou",

        "nee oru loosu da",

        "intha topic konjam explain pannu",

        "thanks da help pannathukku",

        "please help me with this problem",

        "why are you doing this?",

        "nee nalla work pannirukka",
    ]

    for index, message in enumerate(
        test_messages,
        start=1
    ):

        print(
            "\n" + "-" * 75
        )

        print(
            f"TEST MESSAGE {index}"
        )

        print(
            "-" * 75
        )

        print(
            f"Input: {message}"
        )

        try:

            result = pipeline.predict(
                message
            )

            print(
                "\nPrediction Result:"
            )

            for key, value in result.items():

                print(
                    f"{key:20s}: {value}"
                )

        except Exception as error:

            print(
                f"\nPrediction failed: {error}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()