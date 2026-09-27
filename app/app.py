import os
import importlib.util

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

os.chdir(
    BASE_DIR
)


# ============================================================
# LOAD PREDICTION PIPELINE
# ============================================================

PIPELINE_PATH = os.path.join(
    BASE_DIR,
    "src",
    "13_prediction_pipeline.py"
)


spec = importlib.util.spec_from_file_location(
    "prediction_pipeline",
    PIPELINE_PATH
)

if (
    spec is None
    or spec.loader is None
):

    raise ImportError(
        "Could not load prediction pipeline."
    )


prediction_module = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    prediction_module
)


PredictionPipeline = (
    prediction_module.PredictionPipeline
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
)


# ============================================================
# LOAD ML PIPELINE
# ============================================================

try:

    pipeline = PredictionPipeline()

except Exception as error:

    print(
        "Failed to initialize prediction pipeline:"
    )

    print(
        error
    )

    raise


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.get("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# ANALYSIS PAGE
# ============================================================

@app.get("/analysis")
def analysis():

    return render_template(
        "analysis.html"
    )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

@app.get("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({

        "status": "ok",

        "message": (
            "Cyberbullying Detection API "
            "is running"
        ),

        "model": (
            "realworld_multilingual"
        ),

        "pipeline": (
            "TF-IDF + Logistic Regression "
            "+ Linear SVM"
        ),

        "prediction_components": [

            "cyberbullying",

            "offense_category",

            "intent",

            "content_category",

            "severity",

            "target_type",

            "explanation",

            "polite_suggestion",
        ],

        "feature_dimension": 27665,

    }), 200


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict():

    # --------------------------------------------------------
    # GET JSON BODY
    # --------------------------------------------------------

    data = request.get_json(
        silent=True
    )

    if data is None:

        return jsonify({

            "error": (
                "Request body must contain "
                "valid JSON."
            )

        }), 400

    # --------------------------------------------------------
    # CHECK OBJECT TYPE
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict
    ):

        return jsonify({

            "error": (
                "JSON body must be an object."
            )

        }), 400

    # --------------------------------------------------------
    # CHECK TEXT FIELD
    # --------------------------------------------------------

    if "text" not in data:

        return jsonify({

            "error": (
                "Missing required field: text"
            )

        }), 400

    text = data["text"]

    # --------------------------------------------------------
    # VALIDATE TEXT TYPE
    # --------------------------------------------------------

    if not isinstance(
        text,
        str
    ):

        return jsonify({

            "error": (
                "text must be a string"
            )

        }), 400

    # --------------------------------------------------------
    # VALIDATE EMPTY TEXT
    # --------------------------------------------------------

    text = text.strip()

    if not text:

        return jsonify({

            "error": (
                "text cannot be empty"
            )

        }), 400

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        result = pipeline.predict(
            text
        )

        return jsonify(
            result
        ), 200

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    except (
        TypeError,
        ValueError
    ) as error:

        return jsonify({

            "error": (
                "Invalid input"
            ),

            "details": str(
                error
            ),

        }), 400

    # --------------------------------------------------------
    # SERVER / MODEL ERROR
    # --------------------------------------------------------

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        return jsonify({

            "error": (
                "Prediction failed"
            ),

            "details": str(
                error
            ),

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "CYBERBULLYING DETECTION SYSTEM"
    )

    print(
        "=" * 75
    )

    print(
        "Model       : Real-World Multilingual"
    )

    print(
        "Features    : 27,665"
    )

    print(
        "Dashboard   : http://127.0.0.1:5000/"
    )

    print(
        "Analysis    : http://127.0.0.1:5000/analysis"
    )

    print(
        "About       : http://127.0.0.1:5000/about"
    )

    print(
        "Health      : http://127.0.0.1:5000/health"
    )

    print(
        "Prediction  : POST /predict"
    )

    print(
        "=" * 75
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )