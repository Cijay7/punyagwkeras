"""Tes GRU memakai data sintetis; tidak mengakses MySQL."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ews_gambut import FEATURES, HORIZON, LAGS, predictors, supervised
from ews_gambut_pytorch import (ParameterGRU, TrainingConfig, fit_gru,
                                forecast_device, load_checkpoint,
                                parameter_samples, predict_gru, save_checkpoint,
                                sequence_input, temporal_split, train_pytorch)


class PytorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        index = pd.date_range("2026-01-01", periods=400, freq="h")
        cls.hourly = pd.DataFrame({s: v + np.sin(np.arange(400) / 8)
                                  for s, v in zip(FEATURES, [60, 29, 5, 2, 70, 3])}, index=index)

    def test_windows_keep_devices_separate_and_time_split_has_gap(self):
        other = self.hourly.copy() + 100
        other["RR"] = np.nan
        x, y = parameter_samples({"a": self.hourly, "b": other}, "SM")
        self.assertEqual(len(x), 2 * (400 - LAGS - HORIZON + 1))
        first_b = x.loc["b"].iloc[0]
        self.assertEqual(first_b.SM_lag23, other.SM.iloc[0])
        self.assertEqual(y.loc["b"].iloc[0].SM_h24, other.SM.iloc[47])
        masks, meta = temporal_split(x)
        times = x.index.get_level_values("date_time")
        self.assertLess((times[masks["train"]] + pd.Timedelta(hours=24)).max(), pd.Timestamp(meta["validation_start"]))
        self.assertLess((times[masks["validation"]] + pd.Timedelta(hours=24)).max(), pd.Timestamp(meta["test_start"]))
        rr, _ = parameter_samples({"a": self.hourly, "b": other}, "RR")
        self.assertEqual(set(rr.index.get_level_values("device_id")), {"a"})

    def test_sequence_is_oldest_to_newest_and_uses_only_past(self):
        frame = self.hourly[["ST"]].copy()
        x = predictors(frame).iloc[[30]]
        sequence, level = sequence_input(x, "ST", 2.0)
        self.assertEqual(tuple(sequence.shape), (1, LAGS, 3))
        self.assertEqual(sequence[0, -1, 0], 0)
        self.assertAlmostEqual(float(sequence[0, 0, 0]), (frame.ST.iloc[7] - frame.ST.iloc[30]) / 2, places=6)
        self.assertAlmostEqual(level[0, 0], frame.ST.iloc[30], places=5)
        frame.iloc[31:] = 9999
        altered, _ = sequence_input(predictors(frame).iloc[[30]], "ST", 2.0)
        torch.testing.assert_close(sequence, altered)

    def test_scaler_is_fit_only_on_train_and_checkpoint_roundtrip(self):
        x, y = supervised(self.hourly[["ST"]])
        cfg = TrainingConfig(epochs=3, patience=2)
        model, scale, epoch, history = fit_gru(x.iloc[:80], y.iloc[:80], "ST", cfg,
                                              (x.iloc[80:100], y.iloc[80:100] + 10000))
        expected = np.std(y.iloc[:80].to_numpy() - x.iloc[:80].ST_lag0.to_numpy()[:, None])
        self.assertAlmostEqual(scale, expected)
        self.assertGreaterEqual(epoch, 1)
        meta = {"sensor": "ST", "lag_hours": LAGS, "horizon_hours": HORIZON, "framework": "pytorch"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "st.pt"
            save_checkpoint(path, model, scale, meta, cfg)
            loaded, loaded_scale, _ = load_checkpoint(path)
            np.testing.assert_array_equal(predict_gru(model, x.iloc[-3:], "ST", scale),
                                          predict_gru(loaded, x.iloc[-3:], "ST", loaded_scale))

    def test_seed_reproduces_training(self):
        x, y = supervised(self.hourly[["ST"]])
        cfg = TrainingConfig(epochs=2)
        first, scale, _, _ = fit_gru(x.iloc[:64], y.iloc[:64], "ST", cfg, fixed_epochs=2)
        second, other_scale, _, _ = fit_gru(x.iloc[:64], y.iloc[:64], "ST", cfg, fixed_epochs=2)
        np.testing.assert_array_equal(predict_gru(first, x.iloc[-2:], "ST", scale),
                                      predict_gru(second, x.iloc[-2:], "ST", other_scale))

    def test_snapshot_with_removed_hours_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gap harus berupa NaN"):
            parameter_samples({"a": self.hourly.drop(self.hourly.index[100])}, "SM")

    def test_six_models_forecast_new_device_and_test_csv_matches_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            bundles, metrics = train_pytorch({"a": self.hourly}, folder, TrainingConfig(epochs=1))
            self.assertEqual(len(list(Path(folder).glob("model_parameter_*.pt"))), 6)
            result = forecast_device(self.hourly.tail(LAGS), bundles, {"enabled": False}, self.hourly.index[-1])
            self.assertEqual(len(result), HORIZON)
            records = pd.read_csv(Path(folder) / "prediksi_test.csv")
            for sensor in FEATURES:
                part = records.loc[records.sensor.eq(sensor)]
                expected = np.abs(part.gru - part.actual).mean()
                self.assertAlmostEqual(expected, metrics.loc[metrics.sensor.eq(sensor), "mae_gru"].mean())
            loaded = {s: load_checkpoint(Path(folder) / f"model_parameter_{s}.pt") for s in FEATURES}
            pd.testing.assert_frame_equal(result, forecast_device(self.hourly.tail(LAGS), loaded,
                                                                   {"enabled": False}, self.hourly.index[-1]))
            with self.assertRaisesRegex(ValueError, "melewati waktu input"):
                forecast_device(self.hourly.iloc[-25:-1], loaded, {"enabled": False}, self.hourly.index[-1])
            gap = self.hourly.tail(LAGS).copy()
            gap.iloc[5, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "tidak lengkap"):
                forecast_device(gap, loaded, {"enabled": False}, self.hourly.index[-1])


if __name__ == "__main__":
    unittest.main()
