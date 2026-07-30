from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .features import extract_features_for_variant, features_to_frame


@dataclass
class ModelBundle:
    path: Path
    variant: str
    model_name: str
    model: object
    label_encoder: object
    feature_columns: list[str]
    metrics: dict
    metadata: dict


def save_model_bundle(
    path,
    *,
    variant,
    model_name,
    model,
    label_encoder,
    feature_columns,
    metrics=None,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle(
        path=path,
        variant=variant,
        model_name=model_name,
        model=model,
        label_encoder=label_encoder,
        feature_columns=list(feature_columns),
        metrics=metrics or {},
        metadata=metadata or {},
    )
    with path.open("wb") as file:
        pickle.dump(bundle, file)
    return bundle


def load_model_bundle(path):
    path = Path(path)
    with path.open("rb") as file:
        bundle = pickle.load(file)

    if isinstance(bundle, dict):
        bundle = ModelBundle(
            path=path,
            variant=bundle["variant"],
            model_name=bundle["model_name"],
            model=bundle["model"],
            label_encoder=bundle["label_encoder"],
            feature_columns=list(bundle["feature_columns"]),
            metrics=bundle.get("metrics", {}),
            metadata=bundle.get("metadata", {}),
        )
    else:
        bundle.path = path
    return bundle


def list_model_bundles(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.pickle"))


def preferred_model_path(model_paths):
    if not model_paths:
        return None

    bundles = [(path, load_model_bundle(path)) for path in model_paths]
    camera_bundles = [
        (path, bundle)
        for path, bundle in bundles
        if bundle.metadata.get("camera_robust") or "camera" in bundle.variant.lower()
    ]
    if camera_bundles:
        return max(camera_bundles, key=lambda item: item[1].metrics.get("f1_weighted", 0))[0]

    return max(bundles, key=lambda item: item[1].metrics.get("f1_weighted", 0))[0]


def predict_from_features(bundle, features):
    frame = features_to_frame(features, bundle.feature_columns)
    prediction = int(bundle.model.predict(frame)[0])
    label = str(bundle.label_encoder.inverse_transform([prediction])[0])

    confidence = None
    top_predictions = []
    if hasattr(bundle.model, "predict_proba"):
        probabilities = np.asarray(bundle.model.predict_proba(frame)[0], dtype=float)
        order = np.argsort(probabilities)[::-1]
        confidence = float(probabilities[prediction])
        for idx in order[:3]:
            top_predictions.append({
                "label": str(bundle.label_encoder.inverse_transform([int(idx)])[0]),
                "confidence": float(probabilities[idx]),
            })

    label, confidence, top_predictions = _apply_camera_disease_sensitivity(
        bundle,
        label,
        confidence,
        top_predictions,
    )

    return {
        "label": label,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


def _apply_camera_disease_sensitivity(bundle, label, confidence, top_predictions):
    variant = (bundle.variant or "").lower()
    if "camera" not in variant and not bundle.metadata.get("camera_robust"):
        return label, confidence, top_predictions
    if "healthy" not in label.lower() or confidence is None:
        return label, confidence, top_predictions

    plant_name = label.split(" - ", 1)[0]
    disease_candidates = [
        item
        for item in top_predictions
        if item["label"].split(" - ", 1)[0] == plant_name
        and "healthy" not in item["label"].lower()
    ]
    if not disease_candidates:
        return label, confidence, top_predictions

    best_disease = max(disease_candidates, key=lambda item: item["confidence"])
    if confidence - best_disease["confidence"] > 0.08:
        return label, confidence, top_predictions

    adjusted = [best_disease] + [item for item in top_predictions if item["label"] != best_disease["label"]]
    return best_disease["label"], best_disease["confidence"], adjusted


def predict_image(bundle, image_path):
    features = extract_features_for_variant(image_path, bundle.variant)
    result = predict_from_features(bundle, features)
    result["model"] = bundle.model_name
    result["variant"] = bundle.variant
    return result


def bundle_summary(path):
    bundle = load_model_bundle(path)
    return {
        "file": Path(path).name,
        "name": bundle.model_name,
        "variant": bundle.variant,
        "classes": [str(value) for value in bundle.label_encoder.classes_],
        "feature_count": len(bundle.feature_columns),
        "metrics": bundle.metrics,
    }
