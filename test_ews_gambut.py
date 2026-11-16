"""Pemeriksaan kebocoran waktu dan aturan risiko; data sintetis hanya untuk tes."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from parameter_models import ParameterModel, select_parameter_model

from ews_gambut import (FEATURES, HORIZON, LAGS, classify_rules, prepare_device,
                        predictors, supervised, train_parameters, predict_parameters, run_all)


class EwsTests(unittest.TestCase):
    def test_features_never_read_future(self):
        index = pd.date_range("2026-01-01", periods=100, freq="h")
        data = pd.DataFrame(np.arange(600).reshape(100, 6), columns=FEATURES, index=index)
        altered = data.copy()
        altered.iloc[51:] = -999
        pd.testing.assert_frame_equal(predictors(data).iloc[:51], predictors(altered).iloc[:51])
        x, y = supervised(data)
        self.assertEqual(y.loc[x.index[0], "SM_h24"], data.loc[x.index[0] + pd.Timedelta(hours=24), "SM"])

    def test_hour_bins_do_not_read_after_origin(self):
        raw = pd.DataFrame({"id": [1, 2], "device_id": ["a", "a"],
                            "date_time": pd.to_datetime(["2026-01-01 00:30", "2026-01-01 01:30"]),
                            **{sensor: [1.0, 999.0] for sensor in FEATURES}})
        hourly, _ = prepare_device(raw, "a")
        self.assertEqual(hourly.iloc[-1].SM, 1)
        self.assertEqual(hourly.index[-1], pd.Timestamp("2026-01-01 01:00"))

    def test_risk_boundaries_missing_and_stale(self):
        # Ambang arbitrer khusus tes, bukan ambang operasional.
        config = {"enabled": True, "source": "synthetic test only",
                  "units_and_reference_confirmed": True,
                  "rules": [{"sensor": "SM", "direction": "low", "rawan": 50, "bahaya": 20}]}
        frame = pd.DataFrame({"SM": [80, 50, 20, np.nan, 80], "data_lama": [False] * 4 + [True]})
        self.assertEqual(classify_rules(frame, config).status.tolist(),
                         ["Normal", "Rawan", "Bahaya", "Data tidak lengkap", "Data lama"])
        disabled = classify_rules(frame, {"enabled": False})
        self.assertEqual(disabled.status.iloc[0], "Belum dikonfigurasi")
        config["rules"][0]["bahaya"] = 60
        with self.assertRaises(ValueError):
            classify_rules(frame, config)


class CandidateTests(unittest.TestCase):
    def test_constant_inputs_stay_constant_for_every_candidate(self):
        index = pd.date_range("2026-01-01", periods=100, freq="h")
        hourly = pd.DataFrame({"SM": np.full(100, 55.0)}, index=index)
        x, y = supervised(hourly)
        model, scores = select_parameter_model(x.iloc[:20], y.iloc[:20], x.iloc[20:], y.iloc[20:], "SM")
        self.assertEqual(model.methods, ["Nilai terakhir"] * HORIZON)
        model.fit(x, y)
        np.testing.assert_array_equal(model.predict(x), np.full((len(x), HORIZON), 55.0))
        self.assertTrue(all(row["mae_per_candidate"]["Nilai terakhir"] == 0 for row in scores))

    def test_seasonal_prediction_aligns_hours_without_future_inputs(self):
        index = pd.date_range("2026-01-01", periods=100, freq="h")
        hourly = pd.DataFrame({"ST": np.arange(100, dtype=float)}, index=index)
        x, y = supervised(hourly)
        model = ParameterModel("ST", ["Musiman 24 jam"] * HORIZON).fit(x, y)
        # asal pertama jam 23 -> prediksi jam 24 menyalin jam 0, jam 47 menyalin jam 23.
        np.testing.assert_array_equal(model.predict(x.iloc[[0]])[0], np.arange(24))


class ParameterTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range("2026-01-01", periods=400, freq="h")
        cls.first = pd.DataFrame({sensor: value + np.sin(np.arange(400) / 12)
                                 for sensor, value in zip(FEATURES, [60, 29, 5, 2, 70, 3])}, index=index)
        cls.second = cls.first + 1
        cls.second["RR"] = np.nan
        cls.devices = {"a": cls.first, "b": cls.second, "new": cls.first.tail(LAGS)}
        cls.bundles, cls.metrics = train_parameters(cls.devices)

    def test_models_are_per_parameter_and_missingness_is_independent(self):
        self.assertEqual(set(self.bundles), set(FEATURES))
        self.assertEqual(len(self.metrics), len(FEATURES) * HORIZON)
        for sensor, bundle in self.bundles.items():
            self.assertEqual(list(bundle["model"].feature_names_in_),
                             [f"{sensor}_lag{lag}" for lag in range(LAGS)] + ["hour_sin", "hour_cos"])
            self.assertEqual(bundle["metadata"]["training_devices"], ["a"] if sensor == "RR" else ["a", "b"])
            self.assertEqual(bundle["metadata"]["trained_until"], str(self.first.index[-1]))
        # Jendela tetap dibentuk per perangkat; dua histori identik menggandakan jumlahnya.
        self.assertEqual(self.bundles["SM"]["metadata"]["train_windows"],
                         2 * self.bundles["RR"]["metadata"]["train_windows"])

    def test_device_without_training_history_can_use_shared_models(self):
        predicted = predict_parameters(self.devices["new"], self.bundles)
        self.assertEqual(predicted.shape, (HORIZON, len(FEATURES)))
        self.assertTrue(np.isfinite(predicted).all())
        with self.assertRaisesRegex(ValueError, "melewati waktu input"):
            predict_parameters(self.first.iloc[-LAGS-1:-1], self.bundles)

    def test_run_all_saves_only_parameter_models_and_forecasts_new_device(self):
        raw = []
        for device, hourly in self.devices.items():
            frame = hourly.reset_index(names="date_time")
            frame["device_id"] = device
            frame["id"] = np.arange(len(frame))
            raw.append(frame)
        with tempfile.TemporaryDirectory() as directory:
            forecasts, metrics, audit = run_all(pd.concat(raw), {"enabled": False}, directory,
                                                as_of=self.first.index[-1])
            self.assertEqual({p.name for p in Path(directory).glob("*.joblib")},
                             {f"model_parameter_{sensor}.joblib" for sensor in FEATURES})
            self.assertEqual(set(forecasts.device_id), {"a", "new"})
            self.assertEqual(len(forecasts), 2 * HORIZON)
            self.assertEqual(len(metrics), len(FEATURES) * HORIZON)
            self.assertIn("RR tidak lengkap", audit.loc[audit.device_id.eq("b"), "result"].iloc[0])


if __name__ == "__main__":
    unittest.main()
