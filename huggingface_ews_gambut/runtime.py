"""Loader inferensi enam champion Keras. Tidak memuat kode training atau database."""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import json
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from core import FEATURES, HORIZON, LAGS, bounded, sequence


class Models:
    def __init__(self, folder, timezone="Asia/Jakarta"):
        self.folder, self.timezone = Path(folder).resolve(), timezone
        self.lock = RLock()
        self.items = {}
        provenance = json.loads((self.folder/"provenance.json").read_text(encoding="utf-8"))
        if timezone != provenance["timezone"]:
            raise ValueError("TIMEZONE harus sama dengan zona waktu training.")
        manifest = json.loads((self.folder/"model_terbaik.json").read_text(encoding="utf-8"))
        import keras
        for sensor in FEATURES:
            record = manifest[sensor]
            expected = self.folder/"champions"/f"model_{sensor}.keras"
            if record.get("status") != "selected" or (self.folder/record["champion_path"]).resolve() != expected:
                raise ValueError("Manifest champion tidak valid.")
            bundle = json.loads(expected.with_suffix(".json").read_text(encoding="utf-8"))
            if hashlib.sha256(expected.read_bytes()).hexdigest() != bundle["model_sha256"]:
                raise ValueError("Checksum checkpoint tidak cocok.")
            meta, scale = bundle["metadata"], bundle["scale"]
            source = record["device_id"]
            if (meta.get("framework") != "keras" or meta.get("sensor") != sensor or
                    meta.get("architecture") != "GRU" or meta.get("training_scope") != "device_parameter" or
                    meta.get("device_id") != source or meta.get("training_devices") != [source] or
                    meta.get("lag_hours") != LAGS or meta.get("horizon_hours") != HORIZON or
                    meta.get("time_semantics") != "synthetic_not_actual_measurement" or
                    not np.isfinite(scale) or scale <= 0 or pd.isna(pd.Timestamp(meta["trained_until"])) or
                    pd.Timestamp(meta["trained_until"]).tzinfo is not None):
                raise ValueError("Metadata champion tidak sesuai.")
            model = keras.models.load_model(expected, compile=False, safe_mode=True)
            if model.input_shape != (None, LAGS, 3) or model.output_shape != (None, HORIZON):
                raise ValueError("Dimensi Keras tidak sesuai training.")
            self.items[sensor] = {"model": model, "scale": scale, "metadata": meta, "selection": record}

    def info(self, origin=None):
        return {sensor: {"source_device_id": item["metadata"]["device_id"], "framework": "keras",
                         "architecture": "GRU", "trained_until": pd.Timestamp(item["metadata"]["trained_until"]).tz_localize(self.timezone).isoformat(),
                         "training_after_input": bool(origin is not None and pd.Timestamp(item["metadata"]["trained_until"]) > origin),
                         **{key: item["selection"].get(key) for key in
                            ("validation_macro_mae", "test_macro_mae", "test_baseline_macro_mae")}}
                for sensor, item in self.items.items()}

    def predict(self, window):
        result = np.empty((HORIZON, len(FEATURES)))
        with self.lock:
            for j, sensor in enumerate(FEATURES):
                item = self.items[sensor]
                inputs, level = sequence(window, sensor, item["scale"])
                delta = np.asarray(item["model"].predict_on_batch(inputs))
                if delta.shape != (1, HORIZON) or not np.isfinite(delta).all():
                    raise ValueError("Keluaran model tidak valid.")
                result[:, j] = bounded(level+delta[0]*item["scale"], sensor)
        if not np.isfinite(result).all():
            raise ValueError("Prediksi tidak finite.")
        return result
