from pathlib import Path
import random
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from plant_classifier.features import extract_camera_features
from plant_classifier.model_store import save_model_bundle


RANDOM_STATE = 42
MODEL_DIR = BASE_DIR / "models"
DATASET_ROOT = BASE_DIR / "archive" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"

SELECTED_CLASSES = {
    "Peach___Bacterial_spot": "Peach - Bacterial spot",
    "Peach___healthy": "Peach - Healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper bell - Bacterial spot",
    "Pepper,_bell___healthy": "Pepper bell - Healthy",
    "Strawberry___Leaf_scorch": "Strawberry - Leaf scorch",
    "Strawberry___healthy": "Strawberry - Healthy",
}


def collect_subset(split_dir, per_class):
    rows = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for folder_name, label in SELECTED_CLASSES.items():
        image_paths = sorted(path for path in (split_dir / folder_name).iterdir() if path.suffix.lower() in valid_ext)
        sampled = random.sample(image_paths, min(per_class, len(image_paths)))
        for image_path in sampled:
            rows.append({"path": image_path, "label": label})
    return pd.DataFrame(rows)


def phone_like_transform(image):
    small = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    scaled = cv2.resize(small, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(scaled, (9, 9), 0)
    return cv2.convertScaleAbs(blurred, alpha=1.18, beta=18)


def augmentations(image):
    return [
        ("original", image),
        ("blur_9x9", cv2.GaussianBlur(image, (9, 9), 0)),
        ("phone_like", phone_like_transform(image)),
    ]


def build_camera_feature_table(df, include_augmented):
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for idx, row in df.reset_index(drop=True).iterrows():
            image = cv2.imread(str(row["path"]))
            if image is None:
                continue

            variants = augmentations(image) if include_augmented else [("original", image)]
            for aug_name, aug_image in variants:
                temp_path = tmp / f"{idx}_{aug_name}.jpg"
                cv2.imwrite(str(temp_path), aug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                feature_row = extract_camera_features(temp_path)
                feature_row["path"] = str(row["path"])
                feature_row["label"] = row["label"]
                feature_row["augmentation"] = aug_name
                rows.append(feature_row)

            if (idx + 1) % 100 == 0 or idx + 1 == len(df):
                print(f"Extracting camera features: {idx + 1}/{len(df)}")

    return pd.DataFrame(rows)


def metrics_for(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    MODEL_DIR.mkdir(exist_ok=True)

    train_df = collect_subset(TRAIN_DIR, per_class=300)
    valid_df = collect_subset(VALID_DIR, per_class=80)
    train_features = build_camera_feature_table(train_df, include_augmented=True)
    valid_features = build_camera_feature_table(valid_df, include_augmented=False)

    metadata_cols = ["path", "label", "augmentation"]
    feature_columns = [col for col in train_features.columns if col not in metadata_cols]
    X_train = train_features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_valid = valid_features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_features["label"])
    y_valid = label_encoder.transform(valid_features["label"])

    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    pred_valid = model.predict(X_valid)

    output_path = MODEL_DIR / "camera_robust_random_forest.pickle"
    save_model_bundle(
        output_path,
        variant="camera",
        model_name="Camera Robust Random Forest",
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        metrics=metrics_for(y_valid, pred_valid),
        metadata={
            "camera_robust": True,
            "random_state": RANDOM_STATE,
            "train_per_class": 300,
            "valid_per_class": 80,
            "augmentations": ["original", "blur_9x9", "phone_like"],
        },
    )
    print("Camera model saved to:", output_path)


if __name__ == "__main__":
    main()
