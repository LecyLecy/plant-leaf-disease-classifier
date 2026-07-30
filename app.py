from pathlib import Path
import tempfile

from flask import Flask, jsonify, render_template, request

from plant_classifier.model_store import (
    bundle_summary,
    list_model_bundles,
    load_model_bundle,
    predict_image,
    preferred_model_path,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def create_app(model_dir=MODEL_DIR):
    app = Flask(__name__)
    app.config["MODEL_DIR"] = Path(model_dir)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/models")
    def api_models():
        model_paths = list_model_bundles(app.config["MODEL_DIR"])
        models = [bundle_summary(path) for path in model_paths]
        default_model = preferred_model_path(model_paths)
        return jsonify({
            "default_model": default_model.name if default_model else None,
            "models": models,
        })

    @app.post("/api/predict")
    def api_predict():
        model_file = request.form.get("model")
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image was provided."}), 400

        model_paths = list_model_bundles(app.config["MODEL_DIR"])
        if not model_paths:
            return jsonify({"error": "No model bundle is available in the models folder."}), 400

        selected_path = app.config["MODEL_DIR"] / model_file if model_file else model_paths[0]
        if selected_path not in model_paths:
            return jsonify({"error": "The selected model was not found."}), 404

        suffix = Path(image_file.filename or "capture.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            image_file.save(temp.name)
            temp_path = Path(temp.name)

        try:
            bundle = load_model_bundle(selected_path)
            result = predict_image(bundle, temp_path)
            return jsonify(result)
        finally:
            temp_path.unlink(missing_ok=True)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
