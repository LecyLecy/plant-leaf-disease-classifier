import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class DummyModel:
    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


class DummyEncoder:
    classes_ = np.array(["Healthy", "Sick"])

    def inverse_transform(self, values):
        return self.classes_[values]


class PepperEncoder:
    classes_ = np.array(["Pepper bell - Bacterial spot", "Pepper bell - Healthy"])

    def inverse_transform(self, values):
        return self.classes_[values]


class BorderlinePepperModel:
    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.47, 0.51]])


class ModelBundleTests(unittest.TestCase):
    def test_saved_bundle_loads_through_portable_path_unpickler(self):
        from plant_classifier.model_store import load_model_bundle, save_model_bundle

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "portable.pickle"
            save_model_bundle(
                bundle_path,
                variant="fast",
                model_name="Portable",
                model=DummyModel(),
                label_encoder=DummyEncoder(),
                feature_columns=["a"],
            )

            loaded = load_model_bundle(bundle_path)

        self.assertEqual(loaded.model_name, "Portable")
        self.assertEqual(loaded.path, bundle_path)

    def test_predict_from_features_reorders_columns_and_reports_confidence(self):
        from plant_classifier.model_store import ModelBundle, predict_from_features

        bundle = ModelBundle(
            path=Path("dummy.pickle"),
            variant="fast",
            model_name="Dummy",
            model=DummyModel(),
            label_encoder=DummyEncoder(),
            feature_columns=["a", "b", "c"],
            metrics={},
            metadata={},
        )

        result = predict_from_features(bundle, {"c": 3, "a": 1})

        self.assertEqual(result["label"], "Sick")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["top_predictions"][0]["label"], "Sick")

    def test_camera_bundle_prefers_nearby_disease_over_borderline_healthy(self):
        from plant_classifier.model_store import ModelBundle, predict_from_features

        bundle = ModelBundle(
            path=Path("dummy.pickle"),
            variant="camera",
            model_name="Camera",
            model=BorderlinePepperModel(),
            label_encoder=PepperEncoder(),
            feature_columns=["a"],
            metrics={},
            metadata={"camera_robust": True},
        )

        result = predict_from_features(bundle, {"a": 1})

        self.assertEqual(result["label"], "Pepper bell - Bacterial spot")


class FlaskAppTests(unittest.TestCase):
    def test_health_endpoint_reports_available_model(self):
        from app import create_app
        from plant_classifier.model_store import save_model_bundle

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            save_model_bundle(
                model_dir / "dummy.pickle",
                variant="fast",
                model_name="Dummy",
                model=DummyModel(),
                label_encoder=DummyEncoder(),
                feature_columns=["leaf_area_ratio"],
                metrics={"accuracy": 1.0},
                metadata={"test": True},
            )

            app = create_app(model_dir=model_dir)
            response = app.test_client().get("/api/models")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["default_model"], "dummy.pickle")
            self.assertEqual(payload["models"][0]["name"], "Dummy")

    def test_camera_model_is_preferred_as_default(self):
        from app import create_app
        from plant_classifier.model_store import save_model_bundle

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            save_model_bundle(
                model_dir / "full_xgboost.pickle",
                variant="full",
                model_name="Full XGBoost",
                model=DummyModel(),
                label_encoder=DummyEncoder(),
                feature_columns=["leaf_area_ratio"],
                metrics={"f1_weighted": 0.99},
                metadata={},
            )
            save_model_bundle(
                model_dir / "zz_camera_robust_random_forest.pickle",
                variant="fast",
                model_name="Camera Robust Random Forest",
                model=DummyModel(),
                label_encoder=DummyEncoder(),
                feature_columns=["leaf_area_ratio"],
                metrics={"f1_weighted": 0.90},
                metadata={"camera_robust": True},
            )

            app = create_app(model_dir=model_dir)
            response = app.test_client().get("/api/models")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["default_model"], "zz_camera_robust_random_forest.pickle")


if __name__ == "__main__":
    unittest.main()
