import unittest
import pandas as pd
from salin_database_sintetis import identifier, reconstruct, verify_copy

class SyntheticCopyTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame({"id": [9, 1, 5, 2, 10], "device_id": ["a", "a", "b", "b", "b"],
                                    "date_time": pd.to_datetime(["2106-02-07"] * 5),
                                    "unixtime": [4294944000] * 5,
                                    "created_at": pd.to_datetime(["2026-04-13"] * 5),
                                    "soil_moisture": [1.25, 2.5, 0.0, None, 6.5]})

    def test_per_device_counter_is_independent_and_originals_preserved(self):
        result = reconstruct(self.source)
        self.assertEqual(result.id.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(result.original_id.tolist(), [1, 2, 5, 9, 10])
        self.assertEqual(result.device_sequence.tolist(), [0, 0, 1, 1, 2])
        self.assertEqual(result.date_time.tolist(), list(pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 00:10", "2026-01-01 00:20"])))
        self.assertEqual(pd.Timestamp(result.unixtime.iloc[0], unit="s", tz="UTC"), pd.Timestamp("2025-12-31 17:00", tz="UTC"))
        self.assertEqual(verify_copy(self.source, result, "Asia/Jakarta", 600)["rows"], 5)

    def test_verification_detects_changed_sensor_and_wrong_timestamp(self):
        result = reconstruct(self.source)
        result.loc[0, "soil_moisture"] = 999
        with self.assertRaises(AssertionError):
            verify_copy(self.source, result, "Asia/Jakarta", 600)
        result = reconstruct(self.source)
        result.loc[0, "unixtime"] += 60
        with self.assertRaises(AssertionError):
            verify_copy(self.source, result, "Asia/Jakarta", 600)

    def test_bad_ids_missing_devices_and_unsafe_database_names_rejected(self):
        with self.assertRaises(ValueError):
            identifier("database; DROP DATABASE source")
        data = self.source.copy()
        data.loc[0, "device_id"] = None
        with self.assertRaises(ValueError):
            reconstruct(data)
        data = self.source.copy()
        data.loc[0, "id"] = data.id.iloc[1]
        with self.assertRaises(ValueError):
            reconstruct(data)


if __name__ == "__main__":
    unittest.main()
