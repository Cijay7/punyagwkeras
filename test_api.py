"""Tes API memakai data sintetis; tidak mengubah MySQL pengguna."""
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sqlalchemy.exc import OperationalError

from api import Settings, create_app
from ews_gambut import FEATURES, HORIZON, LAGS, supervised
from parameter_models import ParameterModel


class FakeRepository:
    def __init__(self, frame):
        self.frame = frame
        self.history_calls = 0
        self.database_down = False

    def health(self):
        if self.database_down:
            raise OperationalError("private SQL", {}, Exception("private connection data"))

    def latest(self, device_id, now):
        self.health()
        return self.frame.loc[self.frame.device_id.eq(device_id) & self.frame.date_time.le(now)].sort_values(
            ["date_time", "id"], ascending=False).head(1)

    def devices(self, now):
        return self.frame.loc[self.frame.date_time.le(now)].groupby("device_id").date_time.max().reset_index(name="latest_time")

    def history(self, device_id, row):
        self.history_calls += 1
        return self.frame.loc[self.frame.device_id.eq(device_id) & self.frame.date_time.le(row.date_time)].copy()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rules_path = self.root / "rules.json"
        self.rules_path.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        self.settings = Settings("unused", self.root, self.rules_path)
        self.now = pd.Timestamp("2026-11-13 12:30:00")
        times = pd.date_range(end="2026-11-13 12:00:00", periods=90, freq="h")
        hourly = pd.DataFrame({sensor: np.ones(len(times)) * value
                               for sensor, value in zip(FEATURES, [65, 29, 5, 2, 70, 0])}, index=times)
        self.bundles = {}
        self.model_paths = {}
        for sensor in FEATURES:
            x, y = supervised(hourly[[sensor]])
            self.bundles[sensor] = {"model": Ridge().fit(x, y), "metadata": {
                "training_scope": "parameter", "sensor": sensor,
                "trained_until": "2026-11-10 12:00:00", "lag_hours": LAGS,
                "horizon_hours": HORIZON, "method": "Nilai terakhir"}}
            self.model_paths[sensor] = self.root / f"model_parameter_{sensor}.joblib"
            joblib.dump(self.bundles[sensor], self.model_paths[sensor])
        self.bundle = self.bundles["SM"]
        self.model_path = self.model_paths["SM"]
        frame = hourly.reset_index(names="date_time")
        frame["id"] = np.arange(1, len(frame) + 1)
        frame["device_id"] = "test-device"
        self.repository = FakeRepository(frame)
        self.client = TestClient(create_app(self.settings, self.repository, clock=lambda: self.now))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def get(self, endpoint="results"):
        return self.client.get(f"/api/v1/devices/test-device/{endpoint}")

    def test_new_database_reading_is_seen_next_request(self):
        before = self.get()
        self.assertEqual(before.status_code, 200)
        self.assertEqual(before.json()["prediction"]["availability"], "ready")
        self.assertEqual(len(before.json()["prediction"]["points"]), 24)
        self.assertEqual(before.headers["cache-control"], "no-store")
        new = self.repository.frame.iloc[-1].copy()
        new["id"] += 1
        new["date_time"] = self.now
        new["SM"] = 30
        self.repository.frame = pd.concat([self.repository.frame, new.to_frame().T], ignore_index=True)
        self.repository.frame["date_time"] = pd.to_datetime(self.repository.frame.date_time)
        after = self.get("latest").json()
        self.assertEqual(after["latest"]["sensors"]["SM"], 30)
        self.assertGreater(after["latest"]["id"], before.json()["latest"]["id"])
        # Jam input lengkap yang berubah harus membatalkan cache prediksi.
        mask = self.repository.frame.date_time.eq(pd.Timestamp("2026-11-13 12:00"))
        self.repository.frame.loc[mask, "SM"] = 40
        next_prediction = self.get().json()["prediction"]["points"][0]
        self.assertEqual(next_prediction["sensors"]["SM"], 40)

    def test_rules_reload_without_restart(self):
        self.assertEqual(self.get("classification").json()["status"], "Belum dikonfigurasi")
        self.rules_path.write_text(json.dumps({"enabled": True, "source": "synthetic test only",
            "units_and_reference_confirmed": True,
            "rules": [{"sensor": "SM", "direction": "low", "rawan": 70, "bahaya": 20}]}), encoding="utf-8")
        self.assertEqual(self.get("classification").json()["status"], "Rawan")
        self.assertEqual(self.get().json()["prediction"]["points"][0]["classification"]["status"], "Rawan")

    def test_stale_data_never_produces_current_forecast(self):
        self.now += pd.Timedelta(hours=5)
        result = self.get().json()
        self.assertEqual(result["prediction"]["availability"], "stale_data")
        self.assertEqual(result["latest"]["classification"]["status"], "Data lama")
        self.assertEqual(result["prediction"]["points"], [])
        self.assertEqual(self.repository.history_calls, 0)

    def test_missing_history_and_model_have_explicit_status(self):
        self.repository.frame = self.repository.frame.iloc[-10:].copy()
        self.assertEqual(self.get().json()["prediction"]["availability"], "insufficient_history")
        self.model_path.unlink()
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_not_found")

    def test_latest_zero_packet_is_not_normal(self):
        self.repository.frame.loc[self.repository.frame.index[-1], ["SM", "ST", "ph", "SEC"]] = 0
        result = self.get().json()
        self.assertEqual(result["prediction"]["availability"], "invalid_latest_reading")
        self.assertEqual(result["latest"]["classification"]["status"], "Data tidak valid")

    def test_future_measurements_are_ignored(self):
        new = self.repository.frame.iloc[-1].copy()
        new["date_time"] = self.now + pd.Timedelta(days=100)
        new["id"] = 999
        self.repository.frame = pd.concat([self.repository.frame, new.to_frame().T], ignore_index=True)
        self.repository.frame["date_time"] = pd.to_datetime(self.repository.frame.date_time)
        self.assertNotEqual(self.get("latest").json()["latest"]["id"], 999)

    def test_errors_do_not_expose_connection_details(self):
        self.repository.database_down = True
        response = self.get()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "database_unavailable")
        self.assertNotIn("private", response.text)
        self.repository.database_down = False
        self.assertEqual(self.client.get("/api/v1/devices/unknown/latest").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/devices/bad.id/results").status_code, 422)

    def test_invalid_config_returns_actionable_error(self):
        self.rules_path.write_text('{"enabled":true,"rules":[]}', encoding="utf-8")
        response = self.get()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "invalid_rules")

    def test_replaced_model_is_loaded_and_future_training_rejected(self):
        self.assertEqual(self.get().json()["prediction"]["availability"], "ready")
        self.bundle["metadata"]["trained_until"] = "2026-11-14 12:00:00"
        joblib.dump(self.bundle, self.model_path)
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_from_future")

    def test_ridge_parameter_inference_uses_parameter_features(self):
        self.bundles["SM"]["metadata"]["method"] = "Ridge"
        joblib.dump(self.bundles["SM"], self.model_paths["SM"])
        result = self.get().json()["prediction"]
        self.assertEqual(result["availability"], "ready")
        self.assertEqual(result["method_per_sensor"]["SM"], "Ridge")
        self.assertAlmostEqual(result["points"][0]["sensors"]["SM"], 65.0)
        self.assertEqual(result["points"][0]["sensors"]["ST"], 29.0)

    def test_selected_parameter_bundle_survives_serialization_and_api(self):
        sensor = "SM"
        hourly = self.repository.frame.set_index("date_time")[[sensor]]
        x, y = supervised(hourly)
        methods = ["Musiman 24 jam"] * HORIZON
        self.bundles[sensor]["model"] = ParameterModel(sensor, methods).fit(x, y)
        self.bundles[sensor]["metadata"].update(method="Seleksi per horizon", method_per_horizon=methods)
        joblib.dump(self.bundles[sensor], self.model_paths[sensor])
        result = self.get().json()["prediction"]
        self.assertEqual(result["availability"], "ready")
        self.assertEqual(result["method_per_horizon"][sensor], methods)
        self.assertEqual(result["points"][0]["sensors"][sensor], 65.0)

    def test_shared_models_work_for_new_device_and_reload_parameter(self):
        other = self.repository.frame.copy()
        other["device_id"] = "new-device"
        other["SM"] = 42.0
        self.repository.frame = pd.concat([self.repository.frame, other], ignore_index=True)
        result = self.client.get("/api/v1/devices/new-device/results").json()
        self.assertEqual(result["prediction"]["availability"], "ready")
        self.assertEqual(result["prediction"]["points"][0]["sensors"]["SM"], 42)
        self.bundles["RR"]["metadata"]["sensor"] = "SM"
        joblib.dump(self.bundles["RR"], self.model_paths["RR"])
        self.assertEqual(self.get().json()["prediction"]["availability"], "model_invalid")

    def test_missing_sensor_serializes_as_null(self):
        self.repository.frame.loc[self.repository.frame.index[-1], "RR"] = np.nan
        response = self.get("latest")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["latest"]["sensors"]["RR"])


if __name__ == "__main__":
    unittest.main()
