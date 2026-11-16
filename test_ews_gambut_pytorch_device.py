"""Verifikasi training terpisah, ranking validation, dan checkpoint pemenang."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ews_gambut import FEATURES
from ews_gambut_pytorch import TrainingConfig, fit_gru, load_checkpoint
from ews_gambut_pytorch import parameter_samples
from ews_gambut_pytorch_device import (cached_fit, choose_winner, common_score, fit_fingerprint, save_atomic,
                                       forecast_champions, reference_devices,
                                       train_device_parameters)


class DeviceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_winner_uses_validation_excludes_constant_and_ignores_test(self):
        candidates = [
            {"device_id": "constant", "eligible_champion": False, "validation_macro_mae": 0, "test_macro_mae": 0},
            {"device_id": "a", "eligible_champion": True, "validation_macro_mae": 1, "test_macro_mae": 100},
            {"device_id": "b", "eligible_champion": True, "validation_macro_mae": 2, "test_macro_mae": 0},
        ]
        self.assertEqual(choose_winner(candidates)["device_id"], "a")
        candidates[1]["test_macro_mae"], candidates[2]["test_macro_mae"] = 0, 1000
        self.assertEqual(choose_winner(candidates)["device_id"], "a")
        self.assertIsNone(choose_winner([candidates[0]]))

    def test_common_score_weights_devices_equally(self):
        index = pd.MultiIndex.from_tuples([("a", pd.Timestamp("2026-01-01"))] * 3 +
                                         [("b", pd.Timestamp("2026-01-01"))], names=["device_id", "date_time"])
        x = pd.DataFrame({"ST_lag0": [0, 0, 0, 0]}, index=index)
        y = pd.DataFrame(np.zeros((4, 24)), index=index)
        prediction = np.array([[2] * 24] * 3 + [[10] * 24])
        score = common_score(x, y, prediction, "ST", ["a", "b"])
        self.assertEqual(score["macro_mae"], 6)

    def test_resume_rejects_changed_fit_values_and_incomplete_checkpoint(self):
        index = pd.date_range("2026-01-01", periods=100, freq="h")
        frame = pd.DataFrame({"ST": 29 + np.sin(np.arange(100) / 8)}, index=index)
        x, y = parameter_samples({"a": frame}, "ST")
        config = TrainingConfig(epochs=1)
        model, scale, _, _ = fit_gru(x, y, "ST", config, fixed_epochs=1)
        split = {"validation_start": "2026-01-10", "test_start": "2026-01-20"}
        meta = {"sensor": "ST", "framework": "pytorch", "lag_hours": 24, "horizon_hours": 24,
                "device_id": "a", "training_devices": ["a"], "training_scope": "device_parameter",
                "best_epoch": 1, "trained_until": str(index[-1]), "fit_fingerprint": fit_fingerprint(x, y), **split}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_atomic(path, model, scale, meta, config)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            self.assertIsNotNone(cached_fit(path, x, y, "ST", "a", config, split, True, fixed_epochs=1))
            changed = x.copy()
            for column in changed.filter(like="ST_lag"):
                changed[column] += 1
            self.assertIsNone(cached_fit(path, changed, y + 1, "ST", "a", config, split, True, fixed_epochs=1))
            path.write_bytes(b"incomplete checkpoint")
            self.assertIsNone(cached_fit(path, x, y, "ST", "a", config, split, True, fixed_epochs=1))

    def test_training_never_mixes_devices_and_champions_can_be_reloaded(self):
        index = pd.date_range("2026-01-01", periods=400, freq="h")
        first = pd.DataFrame({sensor: value + np.sin(np.arange(400) / 8)
                              for sensor, value in zip(FEATURES, [60, 29, 5, 2, 70, 3])}, index=index)
        second = first + .5
        second["SM"] = 70  # model tersimpan, tetapi bukan kandidat pemenang.
        hourly = {"a": first, "b": second, "short": first.tail(24)}
        calls = []

        def track(x, y, *args, **kwargs):
            trained_devices = set(x.index.get_level_values("device_id"))
            self.assertEqual(len(trained_devices), 1)
            calls.append(trained_devices)
            return fit_gru(x, y, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("ews_gambut_pytorch_device.fit_gru", side_effect=track):
            manifest, ranking = train_device_parameters(hourly, directory, TrainingConfig(epochs=1), min_train_windows=48)
            self.assertEqual(len(calls), 36)  # 12 pasangan, masing-masing train/seleksi, evaluasi, final.
            self.assertEqual(len(ranking), 12)
            self.assertEqual(int(ranking.is_champion.sum()), 6)
            self.assertEqual(manifest["SM"]["device_id"], "a")
            for sensor in FEATURES:
                model, scale, meta = load_checkpoint(Path(directory) / manifest[sensor]["champion_path"])
                self.assertEqual(meta["training_devices"], [manifest[sensor]["device_id"]])
                self.assertEqual(meta["training_scope"], "device_parameter")
                self.assertFalse(meta["selection_uses_test"])
            forecast = forecast_champions(first.tail(24), directory, {"enabled": False}, index[-1])
            self.assertEqual(len(forecast), 24)
            self.assertTrue(np.isfinite(forecast[FEATURES].to_numpy()).all())
            audit = pd.read_csv(Path(directory) / "audit_device_parameter.csv")
            self.assertTrue(audit.loc[audit.device_id.eq("short"), "result"].str.contains("tidak cukup").all())
            self.assertFalse(ranking.loc[ranking.device_id.eq("b") & ranking.sensor.eq("SM"), "eligible_champion"].iloc[0])
            # Simulasikan crash setelah validation/evaluasi tersimpan tetapi final belum ada.
            (Path(directory) / "devices" / "a" / "model_parameter_SM.pt").unlink()
            calls.clear()
            resumed_manifest, resumed = train_device_parameters(hourly, directory, TrainingConfig(epochs=1),
                                                                 min_train_windows=48, resume=True)
            self.assertEqual(len(calls), 1)
            pd.testing.assert_frame_equal(ranking.reset_index(drop=True), resumed.reset_index(drop=True))
            self.assertEqual(manifest, resumed_manifest)
            with patch("ews_gambut_pytorch_device.fit_gru", side_effect=AssertionError("Checkpoint tidak dipakai")):
                train_device_parameters(hourly, directory, TrainingConfig(epochs=1), min_train_windows=48, resume=True)


if __name__ == "__main__":
    unittest.main()
