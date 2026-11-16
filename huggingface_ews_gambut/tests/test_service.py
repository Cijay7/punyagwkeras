import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import create_app
from core import FEATURES, History, Unavailable, prepare
from source import SameOriginRedirect, Source, SourceError

ROOT = Path(__file__).resolve().parents[1]


class FakeModels:
    def info(self, origin=None):
        return {s: {"source_device_id": "winner-"+s, "trained_until": "2026-05-14T00:00:00+07:00",
                    "training_after_input": bool(origin is not None and origin<pd.Timestamp("2026-05-14"))} for s in FEATURES}

    def predict(self, window):
        return np.repeat(window.iloc[[-1]].to_numpy(), 24, axis=0)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT/"examples"/"readings.json").read_text())
        self.device = self.payload["device_id"]
        self.provider = Source(example=True, example_path=ROOT/"examples"/"readings.json")
        self.client = TestClient(create_app(FakeModels(), self.provider, lambda: pd.Timestamp("2026-11-15"), api_key="test-key"))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"X-API-Key": "test-key"}

    def test_get_demo_and_model_metadata(self):
        response = self.client.get(f"/api/v1/devices/{self.device}/predictions", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        result = response.json()["prediction"]
        self.assertEqual(result["availability"], "demo_ready")
        self.assertEqual(len(result["points"]), 24)
        self.assertTrue(result["model_training_after_input"])
        self.assertFalse(result["is_backtest"])
        self.assertFalse(result["is_operational"])
        self.assertEqual(result["model_per_parameter"]["SM"]["source_device_id"], "winner-SM")
        self.assertTrue(result["points"][0]["date_time"].endswith("+07:00"))
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_post_is_stateless_and_validates_device_match(self):
        response = self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=self.payload)
        self.assertEqual(response.json()["data_source"], "request_json")
        self.assertEqual(response.json()["prediction"]["availability"], "demo_ready")
        self.assertEqual(self.client.post("/api/v1/devices/other/predictions", headers=self.headers, json=self.payload).status_code, 422)

    def test_auth_and_live_guard(self):
        self.assertEqual(self.client.get("/api/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.get(f"/api/v1/devices/{self.device}/predictions?mode=live", headers=self.headers).json()["prediction"]
        self.assertEqual(response["availability"], "stale_data")
        self.assertEqual(response["points"], [])

    def test_short_history_and_invalid_sensor_packet(self):
        body = copy.deepcopy(self.payload)
        body["readings"] = body["readings"][-1:]
        response = self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=body)
        self.assertEqual(response.json()["prediction"]["availability"], "insufficient_history")
        for sensor in ["SM", "ST", "ph", "SEC"]:
            body["readings"][-1][sensor] = 0
        response = self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=body)
        self.assertEqual(response.json()["prediction"]["availability"], "invalid_latest_reading")

    def test_timezone_required_and_nonfinite_rejected(self):
        body = copy.deepcopy(self.payload)
        body["readings"][0]["date_time"] = "2026-05-12 00:00:00"
        self.assertEqual(self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=body).status_code, 422)
        body = copy.deepcopy(self.payload)
        body["readings"][0]["SM"] = "Infinity"
        self.assertEqual(self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=body).status_code, 422)

    def test_request_order_does_not_change_hourly_window(self):
        first, _ = prepare(History.model_validate(self.payload), "Asia/Jakarta", pd.Timestamp("2026-11-15"))
        second = copy.deepcopy(self.payload)
        second["readings"].reverse()
        other, _ = prepare(History.model_validate(second), "Asia/Jakarta", pd.Timestamp("2026-11-15"))
        pd.testing.assert_frame_equal(first, other)

    def test_null_sensor_and_reset_clock_do_not_crash_service(self):
        body = copy.deepcopy(self.payload)
        for reading in body["readings"]:
            reading["RR"] = None
        response = self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=body)
        self.assertEqual(response.json()["prediction"]["availability"], "insufficient_history")
        body["readings"] = [body["readings"][-1]]
        body["readings"][0]["date_time"] = "1970-01-01T00:00:00+07:00"
        response = self.client.post(f"/api/v1/devices/{self.device}/predictions", headers=self.headers, json=body)
        self.assertEqual(response.json()["prediction"]["availability"], "invalid_time")

    def test_missing_source_configuration_is_explicit(self):
        with self.assertRaises(SourceError) as error:
            Source().devices()
        self.assertEqual(error.exception.code, "source_not_configured")

    def test_https_source_contract_token_and_device_match(self):
        source = Source("https://source.example/api/v1", "source-secret")
        with patch("source.urllib.request.build_opener") as mocked:
            mocked.return_value.open.return_value = io.BytesIO(json.dumps(self.payload).encode())
            history = source.history(self.device)
            self.assertEqual(history.device_id, self.device)
            request = mocked.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer source-secret")
            self.assertTrue(request.full_url.endswith("/history?hours=25"))
        wrong = copy.deepcopy(self.payload)
        wrong["device_id"] = "wrong-device"
        with patch("source.urllib.request.build_opener") as mocked:
            mocked.return_value.open.return_value = io.BytesIO(json.dumps(wrong).encode())
            with self.assertRaises(SourceError) as error:
                source.history(self.device)
            self.assertEqual(error.exception.code, "source_device_mismatch")

    def test_insecure_source_and_redirect_are_rejected(self):
        with self.assertRaises(ValueError):
            Source("http://localhost:3306")
        import urllib.request
        request = urllib.request.Request("https://source.example/api/v1/devices", headers={"Authorization": "Bearer private"})
        with self.assertRaises(SourceError):
            SameOriginRedirect().redirect_request(request, None, 302, "", {}, "http://other.example")


if __name__ == "__main__":
    unittest.main()
