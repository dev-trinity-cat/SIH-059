"""
sea_ice_model.py
----------------
Core ML inference module for Antarctic Sea Ice Concentration prediction.

Architecture:  SeasonalResidualSeaIceCNN
Input shape:   (1, 66, 57, 9)  — 7 days of SIC history + sin/cos seasonal channels
Output shape:  (1, 66, 57, 1)  — predicted SIC grid for the target day

Resources are loaded *lazily* on first call to predict_sea_ice() so that
importing this module never crashes even when model files are absent (the
Flask app can return a clear 503 instead of a hard import error).
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "models"    / "seasonal_cnn.keras"
TFLITE_PATH   = BASE_DIR / "models"    / "seasonal_cnn.tflite"
ARTIFACT_DIR  = BASE_DIR / "artifacts"
CONFIG_PATH   = ARTIFACT_DIR / "model_config.json"

# ---------------------------------------------------------------------------
# TFLite model wrapper with .predict() interface
# ---------------------------------------------------------------------------

class TFLiteModelWrapper:
    """Provides a Keras-compatible .predict() interface around a TFLite Interpreter."""

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.input_shape = tuple(self.input_details[0]["shape"])
        self.output_shape = tuple(self.output_details[0]["shape"])

    def predict(self, model_input: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.interpreter.set_tensor(self.input_index, model_input.astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)


def _load_tflite_model(model_path: Path):
    """Load a TFLite model using tflite-runtime or tensorflow.lite."""
    try:
        import tflite_runtime.interpreter as tflite
        logger.info("Using tflite-runtime for inference.")
        interpreter = tflite.Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf
        logger.info("Using tensorflow.lite for inference.")
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    return TFLiteModelWrapper(interpreter)


def _load_keras_model_with_weights(model_path: Path):
    """Reconstruct exact Functional architecture and load weights from seasonal_cnn.keras."""
    import keras
    from keras import layers

    logger.info("Reconstructing SeasonalResidualSeaIceCNN architecture and loading weights...")
    inputs = keras.Input(shape=(66, 57, 9), name="sea_ice_and_season")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d")(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_1")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    pred_change = layers.Conv2D(1, (3, 3), padding="same", activation="linear", name="predicted_ice_change")(x)

    latest_ice = layers.Lambda(lambda z: z[:, :, :, 6], output_shape=(66, 57), name="latest_observed_ice")(inputs)
    change_map = layers.Lambda(lambda z: keras.ops.squeeze(z, axis=-1), output_shape=(66, 57), name="ice_change_map")(pred_change)
    res = layers.Add(name="residual_prediction")([latest_ice, change_map])
    bounded = layers.Lambda(lambda z: keras.ops.clip(z, 0.0, 1.0), output_shape=(66, 57), name="bounded_ice_prediction")(res)

    model = keras.Model(inputs=inputs, outputs=bounded, name="SeasonalResidualSeaIceCNN")
    model.load_weights(str(model_path))
    return model


# ---------------------------------------------------------------------------
# Lazy resource cache
# ---------------------------------------------------------------------------

_resources: dict | None = None   # populated on first call to _load_resources()


def _load_resources() -> dict:
    """Load the model and NumPy artifacts exactly once."""
    global _resources
    if _resources is not None:
        return _resources

    model = None

    # Priority 1: Load lightweight TFLite deployment model (safe, no segfaults, low memory)
    if TFLITE_PATH.exists():
        try:
            logger.info("Loading TFLite model from %s …", TFLITE_PATH)
            model = _load_tflite_model(TFLITE_PATH)
            logger.info("TFLite model loaded successfully. Output shape: %s", model.output_shape)
        except Exception as exc:
            logger.warning("Failed to load TFLite model (%s), will try Keras weights fallback: %s", TFLITE_PATH, exc)

    # Priority 2: Fallback to Keras model weight-loading from seasonal_cnn.keras
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {TFLITE_PATH} or {MODEL_PATH}\n"
                "Ensure seasonal_cnn.tflite or seasonal_cnn.keras is present."
            )
        try:
            logger.info("Loading weights from %s into reconstructed architecture …", MODEL_PATH)
            model = _load_keras_model_with_weights(MODEL_PATH)
            logger.info("Keras weights loaded successfully. Output shape: %s", getattr(model, "output_shape", "(1, 66, 57)"))
        except Exception as exc:
            logger.error("Failed to load model weights: %s", exc)
            raise

    def _load_npy(name: str) -> np.ndarray:
        path = ARTIFACT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing: {path}")
        return np.load(path)

    spatial_mask   = _load_npy("spatial_mask.npy")    # (66, 57) bool/float
    latitude_grid  = _load_npy("latitude_grid.npy")   # (66, 57) float
    longitude_grid = _load_npy("longitude_grid.npy")  # (66, 57) float

    with open(CONFIG_PATH, "r") as f:
        model_config = json.load(f)

    _resources = {
        "model":          model,
        "spatial_mask":   spatial_mask.astype(bool),
        "latitude_grid":  latitude_grid,
        "longitude_grid": longitude_grid,
        "model_config":   model_config,
    }
    logger.info("All ML resources loaded successfully.")
    return _resources


def is_loaded() -> bool:
    """Return True if resources have already been loaded into memory."""
    return _resources is not None


def get_model_config() -> dict:
    """Return the model config JSON without requiring full model loading."""
    if _resources is not None:
        return _resources["model_config"]
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return _load_resources()["model_config"]



# ---------------------------------------------------------------------------
# Seasonal feature helpers
# ---------------------------------------------------------------------------

def create_seasonal_features(
    target_date: str,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel sin/cos day-of-year features.

    Parameters
    ----------
    target_date : str
        ISO date string, e.g. "2024-06-01".
    height, width : int
        Spatial dimensions of the grid (66, 57).

    Returns
    -------
    sin_feature, cos_feature : np.ndarray of shape (height, width)
    """
    date      = np.datetime64(target_date)
    year_start = date.astype("datetime64[Y]")
    day_of_year = (
        int((date - year_start).astype("timedelta64[D]").astype(int)) + 1
    )
    angle = 2.0 * np.pi * (day_of_year - 1) / 365.25

    sin_feature = np.full((height, width), np.sin(angle), dtype=np.float32)
    cos_feature = np.full((height, width), np.cos(angle), dtype=np.float32)
    return sin_feature, cos_feature


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_model_input(
    last_7_days: np.ndarray | list,
    target_date: str,
    height: int = 66,
    width: int = 57,
) -> np.ndarray:
    """
    Build the (1, H, W, 9) float32 tensor expected by the CNN.

    Parameters
    ----------
    last_7_days : array-like of shape (7, H, W)
        Sea-ice concentration for the 7 days preceding target_date.
        Values should be in [0, 1].  NaN → 0.
    target_date : str
        ISO date string for the day to forecast.
    height, width : int
        Expected spatial dimensions (defaults match training config).

    Returns
    -------
    np.ndarray of shape (1, height, width, 9)
    """
    ice_data = np.asarray(last_7_days, dtype=np.float32)
    expected  = (7, height, width)

    if ice_data.shape != expected:
        raise ValueError(
            f"last_7_days must have shape {expected}, got {ice_data.shape}."
        )

    # Replace NaN / inf with 0 (ocean or missing → no ice)
    ice_data = np.nan_to_num(ice_data, nan=0.0, posinf=1.0, neginf=0.0)

    # Seasonal channels  (H, W)
    sin_feat, cos_feat = create_seasonal_features(target_date, height, width)

    # (7, H, W) → (H, W, 7)
    ice_channels = np.transpose(ice_data, (1, 2, 0))

    # Concatenate → (H, W, 9)
    model_input = np.concatenate(
        [ice_channels, sin_feat[..., np.newaxis], cos_feat[..., np.newaxis]],
        axis=-1,
    )

    # Add batch dim → (1, H, W, 9)
    return model_input[np.newaxis, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers — NumPy → JSON-safe
# ---------------------------------------------------------------------------

def _to_python(arr: np.ndarray, *, nan_as_none: bool = True) -> list:
    """Recursively convert a NumPy array to a nested Python list.

    NaN values are converted to ``None`` so they serialise as JSON ``null``.
    """
    if arr.ndim == 0:
        v = arr.item()
        return None if (nan_as_none and isinstance(v, float) and np.isnan(v)) else v

    return [_to_python(row, nan_as_none=nan_as_none) for row in arr]


def _compute_stats(prediction: np.ndarray, mask: np.ndarray) -> dict:
    """Compute summary statistics over the valid (non-masked) ocean cells."""
    valid = prediction[mask]
    valid = valid[~np.isnan(valid)]

    if valid.size == 0:
        return {"mean_concentration": None, "min_concentration": None,
                "max_concentration": None, "ice_coverage_fraction": None,
                "valid_cells": 0}

    ice_threshold = 0.15  # ≥15 % SIC counts as "ice-covered"
    return {
        "mean_concentration":   float(np.mean(valid)),
        "min_concentration":    float(np.min(valid)),
        "max_concentration":    float(np.max(valid)),
        "ice_coverage_fraction": float(np.mean(valid >= ice_threshold)),
        "valid_cells":          int(valid.size),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_sea_ice(
    last_7_days: np.ndarray | list,
    target_date: str,
) -> dict:
    """
    Run the CNN and return a JSON-serializable prediction dict.

    Parameters
    ----------
    last_7_days : array-like of shape (7, 66, 57)
        Sea-ice concentration history (values 0–1, NaN for missing/ocean).
    target_date : str
        ISO date string for the forecast day (e.g. "2024-06-01").

    Returns
    -------
    dict with keys:
        success          – bool
        target_date      – str
        prediction       – 2-D list (66×57), NaN → null
        latitude_grid    – 2-D list (66×57)
        longitude_grid   – 2-D list (66×57)
        mask             – 2-D bool list (True = valid ocean cell)
        stats            – dict of concentration statistics
        model_info       – dict with name and performance metrics
    """
    res = _load_resources()

    model:          object    = res["model"]
    spatial_mask:   np.ndarray = res["spatial_mask"]
    latitude_grid:  np.ndarray = res["latitude_grid"]
    longitude_grid: np.ndarray = res["longitude_grid"]
    model_config:   dict      = res["model_config"]

    # --- Build input tensor ---
    model_input = prepare_model_input(last_7_days, target_date)

    # --- Inference ---
    raw_output = model.predict(model_input, verbose=0)  # (1, H, W, ?) or (1, H, W)

    # Squeeze batch dim; handle both (H, W, 1) and (H, W) outputs
    prediction = raw_output[0]
    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[..., 0]   # (H, W, 1) → (H, W)

    # --- Apply spatial mask (NaN outside valid ocean cells) ---
    prediction = np.where(spatial_mask, prediction, np.nan)

    # --- Clip to physical range ---
    prediction = np.clip(prediction, 0.0, 1.0)
    # Restore NaN after clip (clip converts NaN→NaN safely in NumPy, but
    # let's be explicit for cells that were masked)
    prediction = np.where(spatial_mask, prediction, np.nan)

    # --- Build response ---
    stats = _compute_stats(prediction, spatial_mask)

    return {
        "success":        True,
        "target_date":    str(target_date),
        "prediction":     _to_python(prediction),
        "latitude_grid":  _to_python(latitude_grid, nan_as_none=False),
        "longitude_grid": _to_python(longitude_grid, nan_as_none=False),
        "mask":           spatial_mask.tolist(),
        "stats":          stats,
        "model_info": {
            "name":        model_config.get("model_name"),
            "dataset":     model_config.get("dataset"),
            "region":      model_config.get("region"),
            "performance": model_config.get("model_performance"),
        },
    }