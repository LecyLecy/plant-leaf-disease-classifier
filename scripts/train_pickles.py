from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from plant_classifier.features import extract_fast_features
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
        image_paths = sorted((split_dir / folder_name).glob("*"))
        image_paths = [path for path in image_paths if path.suffix.lower() in valid_ext]
        sampled = random.sample(image_paths, min(per_class, len(image_paths)))
        for image_path in sampled:
            rows.append({"path": image_path, "label": label, "folder_class": folder_name, "split": split_dir.name})
    return pd.DataFrame(rows)


def build_fast_feature_table(df):
    rows = []
    for _, row in df.iterrows():
        feature_row = extract_fast_features(row["path"])
        feature_row["path"] = str(row["path"])
        feature_row["label"] = row["label"]
        rows.append(feature_row)
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


def train_and_save(feature_train, feature_valid, variant, model_prefix, include_xgboost=False):
    metadata_cols = ["path", "label", "folder_class", "split"]
    feature_columns = [col for col in feature_train.columns if col not in metadata_cols]
    X_train = feature_train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_valid = feature_valid[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(feature_train["label"])
    y_valid = label_encoder.transform(feature_valid["label"])
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}

    if variant == "fast":
        rf_model = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=class_weight_dict,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_valid)
    save_model_bundle(
        MODEL_DIR / f"{model_prefix}_random_forest.pickle",
        variant=variant,
        model_name=f"{model_prefix.replace('_', ' ').title()} Random Forest",
        model=rf_model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        metrics=metrics_for(y_valid, rf_pred),
        metadata={"random_state": RANDOM_STATE},
    )

    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            xgb_model = XGBClassifier(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_valid)
            save_model_bundle(
                MODEL_DIR / f"{model_prefix}_xgboost.pickle",
                variant=variant,
                model_name=f"{model_prefix.replace('_', ' ').title()} XGBoost",
                model=xgb_model,
                label_encoder=label_encoder,
                feature_columns=feature_columns,
                metrics=metrics_for(y_valid, xgb_pred),
                metadata={"random_state": RANDOM_STATE},
            )
        except Exception as exc:
            print("XGBoost was skipped:", exc)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Building the fast model bundle...")
    fast_train_df = collect_subset(TRAIN_DIR, per_class=80)
    fast_valid_df = collect_subset(VALID_DIR, per_class=20)
    fast_train_features = build_fast_feature_table(fast_train_df)
    fast_valid_features = build_fast_feature_table(fast_valid_df)
    train_and_save(fast_train_features, fast_valid_features, "fast", "fast", include_xgboost=False)

    print("Building the full model bundle from cached CSV features...")
    full_train = pd.read_csv(BASE_DIR / "train_handcrafted_features.csv")
    full_valid = pd.read_csv(BASE_DIR / "valid_handcrafted_features.csv")
    train_and_save(full_train, full_valid, "full", "full", include_xgboost=True)
    print("Finished. Model bundles are available in:", MODEL_DIR)


if __name__ == "__main__":
    main()
