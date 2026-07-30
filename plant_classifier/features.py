from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


FAST_IMAGE_SIZE = (96, 96)
FULL_IMAGE_SIZE = (224, 224)


def load_fast_image(image_path, image_size=FAST_IMAGE_SIZE):
    image = Image.open(image_path).convert("RGB").resize(image_size)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    return image


def _fast_arrays(image):
    rgb = np.asarray(image).astype(np.float32)
    hsv = np.asarray(image.convert("HSV")).astype(np.float32)
    gray = np.asarray(image.convert("L")).astype(np.float32)
    return rgb, hsv, gray


def _fast_masks(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    leaf_mask = (s > 25) & (v > 35)
    reddish = (h < 18) | (h > 238)
    yellow_brown = (h > 18) & (h < 55)
    dark_spot = v < 120
    saturated_damage = s > 55
    spot_mask = leaf_mask & saturated_damage & (dark_spot | reddish | yellow_brown)
    return leaf_mask, spot_mask


def _camera_fast_masks(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    color_mask = (s > 25) & (v > 35)
    greenish = (h > 35) & (h < 145)
    leaf_mask = color_mask & greenish
    if leaf_mask.sum() < leaf_mask.size * 0.03:
        leaf_mask = color_mask

    reddish = (h < 18) | (h > 238)
    yellow_brown = (h > 18) & (h < 55)
    dark_spot = v < 120
    saturated_damage = s > 55
    spot_mask = leaf_mask & saturated_damage & (dark_spot | reddish | yellow_brown)
    return leaf_mask, spot_mask


def extract_fast_features(image_path):
    image = load_fast_image(image_path)
    rgb, hsv, gray = _fast_arrays(image)
    leaf_mask, spot_mask = _fast_masks(hsv)

    if leaf_mask.sum() == 0:
        leaf_mask = np.ones(leaf_mask.shape, dtype=bool)

    features = {}
    leaf_pixels = leaf_mask.sum()
    spot_pixels = spot_mask.sum()

    features["leaf_area_ratio"] = float(leaf_pixels / leaf_mask.size)
    features["spot_area_ratio"] = float(spot_pixels / max(leaf_pixels, 1))
    features["dark_leaf_ratio"] = float(((gray < 100) & leaf_mask).sum() / max(leaf_pixels, 1))

    for name, arr in [("rgb", rgb), ("hsv", hsv)]:
        for channel in range(3):
            values = arr[:, :, channel][leaf_mask]
            features[f"{name}_{channel}_mean"] = float(values.mean())
            features[f"{name}_{channel}_std"] = float(values.std())
            features[f"{name}_{channel}_p25"] = float(np.percentile(values, 25))
            features[f"{name}_{channel}_p75"] = float(np.percentile(values, 75))

    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    features["texture_grad_mean"] = float((grad_x + grad_y) / 2)
    features["texture_grad_x"] = float(grad_x)
    features["texture_grad_y"] = float(grad_y)

    hue_values = hsv[:, :, 0][leaf_mask]
    hue_hist, _ = np.histogram(hue_values, bins=8, range=(0, 256), density=True)
    for idx, value in enumerate(hue_hist):
        features[f"hue_hist_{idx}"] = float(value)

    return features


def extract_camera_features(image_path):
    image = load_fast_image(image_path)
    rgb, hsv, gray = _fast_arrays(image)
    leaf_mask, spot_mask = _camera_fast_masks(hsv)

    if leaf_mask.sum() == 0:
        leaf_mask = np.ones(leaf_mask.shape, dtype=bool)

    features = {}
    leaf_pixels = leaf_mask.sum()
    spot_pixels = spot_mask.sum()

    features["leaf_area_ratio"] = float(leaf_pixels / leaf_mask.size)
    features["spot_area_ratio"] = float(spot_pixels / max(leaf_pixels, 1))
    features["dark_leaf_ratio"] = float(((gray < 100) & leaf_mask).sum() / max(leaf_pixels, 1))

    for name, arr in [("rgb", rgb), ("hsv", hsv)]:
        for channel in range(3):
            values = arr[:, :, channel][leaf_mask]
            features[f"{name}_{channel}_mean"] = float(values.mean())
            features[f"{name}_{channel}_std"] = float(values.std())
            features[f"{name}_{channel}_p25"] = float(np.percentile(values, 25))
            features[f"{name}_{channel}_p75"] = float(np.percentile(values, 75))

    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    features["texture_grad_mean"] = float((grad_x + grad_y) / 2)
    features["texture_grad_x"] = float(grad_x)
    features["texture_grad_y"] = float(grad_y)

    hue_values = hsv[:, :, 0][leaf_mask]
    hue_hist, _ = np.histogram(hue_values, bins=8, range=(0, 256), density=True)
    for idx, value in enumerate(hue_hist):
        features[f"hue_hist_{idx}"] = float(value)

    return features


def read_full_image(image_path, image_size=FULL_IMAGE_SIZE):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"The image could not be read: {image_path}")
    image_bgr = cv2.resize(image_bgr, image_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_full_image(image_rgb):
    gaussian = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    denoised = cv2.medianBlur(gaussian, 3)
    return {
        "rgb": image_rgb,
        "denoised": denoised,
        "hsv": cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV),
        "lab": cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB),
        "gray": cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY),
    }


def create_leaf_mask(hsv):
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 25) & (value > 35)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def create_camera_leaf_mask(hsv):
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = (saturation > 25) & (value > 35)
    greenish = (hue > 25) & (hue < 100)
    leaf_mask = color_mask & greenish

    if leaf_mask.sum() < leaf_mask.size * 0.03:
        leaf_mask = color_mask

    mask = leaf_mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def otsu_spot_mask(lab, leaf_mask):
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    _, otsu_a = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_b = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = cv2.bitwise_or(otsu_a, otsu_b)
    return cv2.bitwise_and(otsu_mask, leaf_mask)


def kmeans_spot_mask(lab, leaf_mask, k=3):
    leaf_pixels = leaf_mask > 0
    if leaf_pixels.sum() < 20:
        return np.zeros(leaf_mask.shape, dtype=np.uint8)

    ab_pixels = lab[:, :, 1:3][leaf_pixels].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(ab_pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    disease_cluster = int(np.argmax(centers[:, 0] + centers[:, 1]))

    result = np.zeros(leaf_mask.shape, dtype=np.uint8)
    result[leaf_pixels] = (labels.flatten() == disease_cluster).astype(np.uint8) * 255
    return result


def segment_disease_spots(processed):
    leaf_mask = create_leaf_mask(processed["hsv"])
    otsu_mask = otsu_spot_mask(processed["lab"], leaf_mask)
    kmask = kmeans_spot_mask(processed["lab"], leaf_mask, k=3)

    combined = cv2.bitwise_or(otsu_mask, kmask)
    combined = cv2.bitwise_and(combined, leaf_mask)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return leaf_mask, combined


def safe_skew(values):
    values = values.astype(np.float32)
    std = values.std()
    if std < 1e-6:
        return 0.0
    centered = values - values.mean()
    return float(np.mean((centered / std) ** 3))


def color_moments(image, mask, prefix):
    features = {}
    valid = mask > 0
    if valid.sum() == 0:
        valid = np.ones(mask.shape, dtype=bool)

    for channel_idx in range(image.shape[2]):
        values = image[:, :, channel_idx][valid].astype(np.float32)
        features[f"{prefix}_ch{channel_idx}_mean"] = float(values.mean())
        features[f"{prefix}_ch{channel_idx}_std"] = float(values.std())
        features[f"{prefix}_ch{channel_idx}_skew"] = safe_skew(values)

    return features


def glcm_features(gray, mask, levels=16):
    quantized = np.clip((gray.astype(np.float32) / 256 * levels).astype(np.int32), 0, levels - 1)
    valid = mask > 0
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for dy, dx in offsets:
        y_start = max(0, dy)
        y_end = gray.shape[0] + min(0, dy)
        x_start = max(0, dx)
        x_end = gray.shape[1] + min(0, dx)

        current = quantized[y_start:y_end, x_start:x_end]
        neighbor = quantized[y_start - dy:y_end - dy, x_start - dx:x_end - dx]
        current_mask = valid[y_start:y_end, x_start:x_end]
        neighbor_mask = valid[y_start - dy:y_end - dy, x_start - dx:x_end - dx]
        pair_mask = current_mask & neighbor_mask

        for i, j in zip(current[pair_mask].ravel(), neighbor[pair_mask].ravel()):
            glcm[i, j] += 1

    if glcm.sum() == 0:
        glcm += 1
    glcm /= glcm.sum()

    i_idx, j_idx = np.indices(glcm.shape)
    contrast = np.sum(((i_idx - j_idx) ** 2) * glcm)
    dissimilarity = np.sum(np.abs(i_idx - j_idx) * glcm)
    homogeneity = np.sum(glcm / (1.0 + np.abs(i_idx - j_idx)))
    energy = np.sqrt(np.sum(glcm ** 2))

    mean_i = np.sum(i_idx * glcm)
    mean_j = np.sum(j_idx * glcm)
    std_i = np.sqrt(np.sum(((i_idx - mean_i) ** 2) * glcm))
    std_j = np.sqrt(np.sum(((j_idx - mean_j) ** 2) * glcm))
    correlation = np.sum(((i_idx - mean_i) * (j_idx - mean_j) * glcm) / (std_i * std_j + 1e-8))

    return {
        "glcm_contrast": float(contrast),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation),
    }


def lbp_histogram(gray, mask):
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, :-2], gray[1:-1, :-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    valid = mask[1:-1, 1:-1] > 0
    if valid.sum() == 0:
        valid = np.ones_like(center, dtype=bool)

    hist, _ = np.histogram(lbp[valid], bins=16, range=(0, 256), density=True)
    return {f"lbp_bin_{idx:02d}": float(value) for idx, value in enumerate(hist)}


def shape_features(spot_mask, leaf_mask):
    leaf_area = max(int((leaf_mask > 0).sum()), 1)
    spot_area = int((spot_mask > 0).sum())
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(c) for c in contours]

    return {
        "disease_area_ratio": float(spot_area / leaf_area),
        "spot_count": int(len(contours)),
        "largest_spot_ratio": float((max(contour_areas) if contour_areas else 0.0) / leaf_area),
    }


def extract_full_features(image_path):
    image_rgb = read_full_image(image_path)
    processed = preprocess_full_image(image_rgb)
    leaf_mask, spot_mask = segment_disease_spots(processed)

    features = {}
    features.update(shape_features(spot_mask, leaf_mask))
    features.update(color_moments(processed["hsv"], leaf_mask, "hsv"))
    features.update(color_moments(processed["lab"], leaf_mask, "lab"))
    features.update(glcm_features(processed["gray"], leaf_mask, levels=16))
    features.update(lbp_histogram(processed["gray"], leaf_mask))
    return features


def features_to_frame(features, feature_columns):
    frame = pd.DataFrame([features])
    frame = frame.reindex(columns=feature_columns, fill_value=0)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def extract_features_for_variant(image_path, variant):
    variant = (variant or "").lower()
    if variant == "fast":
        return extract_fast_features(Path(image_path))
    if variant in {"full", "regular"}:
        return extract_full_features(Path(image_path))
    if variant in {"camera", "camera_fast", "camera_robust"}:
        return extract_camera_features(Path(image_path))
    raise ValueError(f"Unknown model variant: {variant}")
