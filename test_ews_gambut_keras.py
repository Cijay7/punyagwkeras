"""Tes Keras memakai data sintetis; tidak mengubah database."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ews_gambut import FEATURES
from ews_gambut_keras import (Config, choose, fit, get_stage, load_bundle, predict,
                              samples, sequence, split_time, validate_synthetic)


class KerasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range("2026-01-01", periods=400, freq="h")
        cls.hourly = pd.DataFrame({s: v + np.sin(np.arange(400) / 8)
                                  for s, v in zip(FEATURES, [60, 29, 5, 2, 70, 3])}, index=index)

    def test_windows_and_time_gap_keep_device_sequences_separate(self):
        other = self.hourly + 1
        other["RR"] = np.nan
        x, y = samples({"a": self.hourly, "b": other}, "ST")
        self.assertEqual(len(x), 2 * (400 - 47))
        self.assertEqual(x.loc["b"].iloc[0].ST_lag23, other.ST.iloc[0])
        masks, split = split_time(x)
        times = x.index.get_level_values("date_time")
        self.assertLess((times[masks["train"]] + pd.Timedelta(hours=24)).max(), pd.Timestamp(split["validation_start"]))
        self.assertLess((times[masks["validation"]] + pd.Timedelta(hours=24)).max(), pd.Timestamp(split["test_start"]))
        rr, _ = samples({"a": self.hourly, "b": other}, "RR")
        self.assertEqual(set(rr.index.get_level_values("device_id")), {"a"})
        with self.assertRaisesRegex(ValueError, "satu device"):
            fit(x, y, "ST", Config(epochs=1))

    def test_sequence_is_oldest_first_and_never_reads_future(self):
        x, _ = samples({"a": self.hourly}, "ST")
        seq, level = sequence(x.iloc[[0]], "ST", 2)
        self.assertEqual(seq.shape, (1, 24, 3))
        self.assertEqual(seq[0, -1, 0], 0)
        self.assertAlmostEqual(float(seq[0, 0, 0]), (self.hourly.ST.iloc[0] - self.hourly.ST.iloc[23]) / 2, places=6)
        altered = self.hourly.copy()
        altered.iloc[24:] = 999
        modified, _ = samples({"a": altered}, "ST")
        np.testing.assert_array_equal(seq, sequence(modified.iloc[[0]], "ST", 2)[0])

    def test_scaler_checkpoint_and_resume(self):
        x, y = samples({"a": self.hourly}, "ST")
        cfg = Config(epochs=2, hidden_size=4, batch_size=64)
        split = {"validation_start": "2026-01-10", "test_start": "2026-01-20"}
        tr, target = x.iloc[:64], y.iloc[:64]
        validation = (x.iloc[64:80], y.iloc[64:80] + 1000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_ST.keras"
            first = get_stage(path, tr, target, "ST", "a", cfg, split, False, validation=validation)
            model, scale, best, history, meta = first
            expected = np.std(target.to_numpy() - tr.ST_lag0.to_numpy()[:, None])
            self.assertAlmostEqual(scale, expected)
            loaded, bundle = load_bundle(path)
            np.testing.assert_allclose(predict(model, x.iloc[-2:], "ST", scale),
                                       predict(loaded, x.iloc[-2:], "ST", bundle["scale"]), rtol=1e-7, atol=1e-7)
            with patch("ews_gambut_keras.fit", side_effect=AssertionError("Tidak boleh fit ulang")):
                resumed = get_stage(path, tr, target, "ST", "a", cfg, split, True, validation=validation)
            self.assertEqual(resumed[2], best)
            self.assertEqual(meta["training_devices"], ["a"])
            path.write_bytes(b"incomplete model")
            with self.assertRaises(ValueError):
                load_bundle(path)

    def test_synthetic_timezone_and_device_counter_are_checked(self):
        raw = pd.DataFrame({"device_id": ["a", "a"], "device_sequence": [0, 1],
                            "date_time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10"]),
                            "unixtime": [1767200400, 1767201000], "time_is_synthetic": [1, 1]})
        meta = {"time_semantics": "synthetic_not_actual_measurement", "timezone": "Asia/Jakarta", "interval_seconds": 600}
        validate_synthetic(raw, meta)
        raw.loc[1, "unixtime"] += 60
        with self.assertRaises(ValueError):
            validate_synthetic(raw, meta)

    def test_winner_uses_validation_and_excludes_constant_candidates(self):
        rows = [{"device_id": "constant", "eligible_champion": False, "validation_macro_mae": 0},
                {"device_id": "a", "eligible_champion": True, "validation_macro_mae": 1, "test_macro_mae": 99},
                {"device_id": "b", "eligible_champion": True, "validation_macro_mae": 2, "test_macro_mae": 0}]
        self.assertEqual(choose(rows)["device_id"], "a")
        rows[1]["test_macro_mae"] = 999
        self.assertEqual(choose(rows)["device_id"], "a")


if __name__ == "__main__":
    unittest.main()
