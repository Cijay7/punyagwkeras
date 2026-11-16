"""Tes HTTP API Keras dengan repository/model palsu; tidak menulis MySQL."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api_keras import KerasSettings, create_app
from ews_gambut import FEATURES


class Repository:
    def __init__(self, frame):
        self.frame = frame
        self.history_calls = 0
        self.down = False
        self.timezone = "Asia/Jakarta"

    def health(self):
        if self.down:
            raise OperationalError("private SQL", {}, Exception("private password"))

    def metadata(self):
        self.health()
        return {"time_semantics": "synthetic_not_actual_measurement", "timezone": self.timezone}

    def latest(self, device, now):
        self.health()
        return self.frame.loc[self.frame.device_id.eq(device) & self.frame.date_time.le(now)].sort_values(["date_time", "id"], ascending=False).head(1)

    def history(self, device, row):
        self.history_calls += 1
        return self.frame.loc[self.frame.device_id.eq(device) & self.frame.date_time.le(row.date_time)].copy()

    def devices(self, now):
        return self.frame.loc[self.frame.date_time.le(now)].groupby("device_id").date_time.max().reset_index(name="latest_time")


class KerasApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "champions").mkdir()
        self.rules = self.root / "rules.json"
        self.rules.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        self.settings = KerasSettings("mysql+pymysql://root@localhost/fake_synthetic", self.root, self.rules)
        self.now = pd.Timestamp("2026-11-15 12:30")
        times = pd.date_range(end="2026-05-13 12:00", periods=30, freq="h")
        frame = pd.DataFrame({"id": range(30), "device_id": "device-a", "date_time": times,
                              "time_is_synthetic": 1, **{s: v for s, v in zip(FEATURES, [65., 29., 5., 2., 70., 0.])}})
        self.repo = Repository(frame)
        manifest = {}
        self.paths = {}
        for index, sensor in enumerate(FEATURES):
            path = self.root / "champions" / f"model_{sensor}.keras"
            path.write_bytes(b"fake-model")
            source = f"winner-{sensor}"
            bundle = {"scale": 1.0, "offset": index + 1, "metadata": {
                "sensor": sensor, "framework": "keras", "architecture": "GRU", "training_scope": "device_parameter",
                "device_id": source, "training_devices": [source], "lag_hours": 24, "horizon_hours": 24,
                "trained_until": "2026-05-14 00:00:00", "time_semantics": "synthetic_not_actual_measurement"}}
            path.with_suffix(".json").write_text(json.dumps(bundle), encoding="utf-8")
            self.paths[sensor] = path
            manifest[sensor] = {"status": "selected", "device_id": source, "champion_path": f"champions/model_{sensor}.keras"}
        self.manifest = manifest
        (self.root / "model_terbaik.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.loads, self.inferences = 0, 0

        def loader(path):
            self.loads += 1
            bundle = json.loads(path.with_suffix(".json").read_text())
            return bundle["offset"], bundle

        def predictor(offset, x, sensor, scale):
            self.inferences += 1
            self.assertEqual(list(x.columns), [f"{sensor}_lag{i}" for i in range(24)] + ["hour_sin", "hour_cos"])
            return np.full((1, 24), x[f"{sensor}_lag0"].iloc[0] + offset)

        self.client = TestClient(create_app(self.settings, self.repo, lambda: self.now, loader, predictor))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def get(self, endpoint="results", **params):
        return self.client.get(f"/api/v1/devices/device-a/{endpoint}", params=params)

    def test_demo_predicts_24_points_with_champion_metadata_and_future_flags(self):
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        prediction = response.json()["prediction"]
        self.assertEqual(prediction["availability"], "demo_ready")
        self.assertTrue(prediction["is_demo"])
        self.assertFalse(prediction["is_backtest"])
        self.assertFalse(prediction["is_operational"])
        self.assertTrue(prediction["model_training_after_input"])
        self.assertTrue(prediction["input_data_lama"])
        self.assertEqual(len(prediction["points"]), 24)
        self.assertEqual(prediction["points"][0]["sensors"]["SM"], 66)
        self.assertEqual(prediction["points"][0]["sensors"]["ST"], 31)
        self.assertEqual(prediction["model_per_parameter"]["SM"]["source_device_id"], "winner-SM")
        self.assertTrue(prediction["points"][0]["date_time"].endswith("+07:00"))
        self.assertEqual(self.loads, 6)

    def test_live_rejects_stale_and_fresh_synthetic_without_loading_models(self):
        self.assertEqual(self.get(mode="live").json()["prediction"]["availability"], "stale_data")
        self.now = pd.Timestamp("2026-05-13 12:30")
        response = self.get(mode="live").json()["prediction"]
        self.assertEqual(response["availability"], "synthetic_data")
        self.assertEqual(response["points"], [])
        self.assertEqual(self.loads, 0)
        self.assertEqual(self.repo.history_calls, 0)

    def test_model_and_prediction_caches_refresh_on_changes(self):
        self.get()
        self.get()
        self.assertEqual((self.loads, self.inferences), (6, 6))
        self.repo.frame.loc[self.repo.frame.index[-1], "SM"] = 40
        self.assertEqual(self.get().json()["prediction"]["points"][0]["sensors"]["SM"], 41)
        self.assertEqual((self.loads, self.inferences), (6, 12))
        path = self.paths["SM"].with_suffix(".json")
        bundle = json.loads(path.read_text())
        bundle["offset"] = 12
        path.write_text(json.dumps(bundle), encoding="utf-8")
        self.assertEqual(self.get().json()["prediction"]["points"][0]["sensors"]["SM"], 52)
        self.assertEqual(self.loads, 12)

    def test_other_device_uses_same_six_champions(self):
        other = self.repo.frame.copy()
        other["device_id"], other["SM"] = "device-b", 30.
        self.repo.frame = pd.concat([self.repo.frame, other], ignore_index=True)
        response = self.client.get("/api/v1/devices/device-b/predictions").json()
        self.assertEqual(response["prediction"]["points"][0]["sensors"]["SM"], 31)
        self.assertEqual(response["prediction"]["model_per_parameter"]["SM"]["source_device_id"], "winner-SM")

    def test_latest_database_reading_and_rules_seen_on_next_request(self):
        self.now = pd.Timestamp("2026-05-13 12:30")
        self.assertEqual(self.get("latest").json()["latest"]["sensors"]["SM"], 65)
        self.repo.frame.loc[self.repo.frame.index[-1], "SM"] = 40
        self.assertEqual(self.get("latest").json()["latest"]["sensors"]["SM"], 40)
        self.rules.write_text(json.dumps({"enabled": True, "source": "synthetic test", "units_and_reference_confirmed": True,
            "rules": [{"sensor": "SM", "direction": "low", "rawan": 50, "bahaya": 20}]}), encoding="utf-8")
        self.assertEqual(self.get("classification").json()["status"], "Rawan")
        self.assertEqual(self.get().json()["prediction"]["points"][0]["classification"]["status"], "Rawan")

    def test_zero_packet_and_incomplete_history_are_explicit(self):
        self.repo.frame.loc[self.repo.frame.index[-1], ["SM", "ST", "ph", "SEC"]] = 0
        self.assertEqual(self.get().json()["prediction"]["availability"], "invalid_latest_reading")
        self.repo.frame = self.repo.frame.iloc[-10:-1].copy()
        self.assertEqual(self.get().json()["prediction"]["availability"], "insufficient_history")

    def test_missing_or_incorrect_model_is_not_used(self):
        self.paths["SM"].unlink()
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_not_found")
        self.paths["SM"].write_bytes(b"fake")
        path = self.paths["SM"].with_suffix(".json")
        bundle = json.loads(path.read_text())
        bundle["metadata"]["device_id"] = "wrong-device"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_invalid")

    def test_manifest_path_cannot_escape_model_directory(self):
        self.manifest["SM"]["champion_path"] = "../private.keras"
        (self.root / "model_terbaik.json").write_text(json.dumps(self.manifest))
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_invalid")

    def test_invalid_metrics_and_nonfinite_inference_are_explicit(self):
        self.manifest["SM"]["test_macro_mae"] = "invalid"
        path = self.root / "model_terbaik.json"
        path.write_text(json.dumps(self.manifest))
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_invalid")
        del self.manifest["SM"]["test_macro_mae"]
        path.write_text(json.dumps(self.manifest))
        self.client.app.state.service.predictor = lambda *args: np.full((1, 24), np.nan)
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_invalid")

    def test_errors_and_input_validation(self):
        self.assertEqual(self.client.get("/api/v1/devices/unknown/latest").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/devices/bad.id/latest").status_code, 422)
        self.assertEqual(self.get(mode="other").status_code, 422)
        self.repo.down = True
        response = self.get()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "database_unavailable")
        self.assertNotIn("private", response.text)

    def test_dataset_timezone_and_flags_are_checked(self):
        self.repo.timezone = "UTC"
        self.assertEqual(self.get().json()["error"]["code"], "dataset_mismatch")
        self.repo.timezone = "Asia/Jakarta"
        self.repo.frame.loc[self.repo.frame.index[-1], "time_is_synthetic"] = 0
        self.assertEqual(self.get().json()["error"]["code"], "dataset_mismatch")

    def test_devices_models_and_health_endpoints(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(len(self.client.get("/api/v1/devices").json()["devices"]), 1)
        self.assertEqual(set(self.client.get("/api/v1/models").json()["models"]), set(FEATURES))
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_null_sensor_serializes_as_null_and_blocks_forecast(self):
        self.repo.frame.loc[self.repo.frame.index[-1], "RR"] = np.nan
        self.assertIsNone(self.get("latest").json()["latest"]["sensors"]["RR"])
        self.assertEqual(self.get().json()["prediction"]["availability"], "insufficient_history")


if __name__ == "__main__":
    unittest.main()
