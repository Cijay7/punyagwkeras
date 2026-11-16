"""Kandidat tetap dan pemilihan validation; tidak membaca test."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BLOCKS = ((0, 6), (6, 12), (12, 24))
CANDIDATES = ("Nilai terakhir", "Musiman 24 jam", "Rata-rata 6 jam",
              "Ridge delta 10", "Ridge delta 100", "Ridge delta 1000",
              "ExtraTrees delta 5", "ExtraTrees delta 20")


def bounded(values, sensor):
    if sensor == "SM":
        return np.clip(values, 0, 100)
    if sensor == "ph":
        return np.clip(values, 0, 14)
    if sensor in ("SEC", "RR"):
        return np.maximum(values, 0)
    return values


class ParameterModel:
    """Prediksi perubahan relatif lag0; satu objek untuk satu parameter."""
    def __init__(self, sensor, methods):
        self.sensor = sensor
        self.methods = list(methods)

    def _features(self, x):
        lag = x[[f"{self.sensor}_lag{i}" for i in range(24)]].to_numpy()
        level = lag[:, [0]]
        relative = lag - level
        features = np.column_stack([level, relative[:, 1:], x[["hour_sin", "hour_cos"]],
                                    relative[:, :6].mean(axis=1), relative.std(axis=1)])
        return lag, features

    def fit(self, x, y):
        self.feature_names_in_ = np.asarray(x.columns)
        lag, features = self._features(x)
        delta = np.asarray(y) - lag[:, [0]]
        self.estimators_ = {}
        for method in set(self.methods):
            if method.startswith("Ridge"):
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=float(method.split()[-1])))
            elif method.startswith("ExtraTrees"):
                estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=int(method.split()[-1]),
                                                max_depth=12, random_state=42, n_jobs=2)
            else:
                continue
            self.estimators_[method] = estimator.fit(features, delta)
        return self

    def candidate_prediction(self, x, method):
        lag, features = self._features(x)
        if method == "Nilai terakhir":
            result = np.repeat(lag[:, [0]], 24, axis=1)
        elif method == "Musiman 24 jam":
            result = lag[:, ::-1].copy()
        elif method == "Rata-rata 6 jam":
            result = np.repeat(lag[:, :6].mean(axis=1)[:, None], 24, axis=1)
        else:
            result = lag[:, [0]] + self.estimators_[method].predict(features)
        # Jangan mengarang variasi untuk histori input yang benar-benar konstan.
        constant = np.ptp(lag, axis=1) == 0
        result[constant] = lag[constant, [0]][:, None]
        return bounded(result, self.sensor)

    def predict(self, x):
        predictions = {method: self.candidate_prediction(x, method) for method in set(self.methods)}
        return np.column_stack([predictions[method][:, h] for h, method in enumerate(self.methods)])


def select_parameter_model(x, y, validation_x, validation_y, sensor):
    """Kandidat dibekukan; ambang peningkatan validation 2% per blok horizon."""
    all_candidates = ParameterModel(sensor, CANDIDATES).fit(x, y)
    actual = np.asarray(validation_y)
    errors = {name: np.abs(all_candidates.candidate_prediction(validation_x, name) - actual)
              for name in CANDIDATES}
    methods, scores = [], []
    for start, end in BLOCKS:
        losses = {name: float(error[:, start:end].mean()) for name, error in errors.items()}
        best = min(losses, key=losses.get)
        if losses[best] >= losses["Nilai terakhir"] * 0.98:
            best = "Nilai terakhir"
        methods.extend([best] * (end - start))
        scores.append({"horizon_start": start + 1, "horizon_end": end,
                       "selected": best, "mae_per_candidate": losses})
    return ParameterModel(sensor, methods), scores
