"""Pastikan audit tidak menyamakan timestamp tidak wajar dengan gagal parsing."""
import tempfile
import unittest

import pandas as pd

from audit_waktu import audit, decode_unix
from ews_gambut import FEATURES


class TimestampAuditTests(unittest.TestCase):
    def test_seconds_milliseconds_microseconds_and_nanoseconds(self):
        epoch = 1776075835
        values = pd.Series([str(epoch * factor) for factor in (1, 1000, 1000000, 1000000000)])
        decoded, units = decode_unix(values)
        self.assertEqual(units.tolist(), ["s", "ms", "us", "ns"])
        for stamp in decoded:
            self.assertLessEqual(abs((stamp - pd.Timestamp(epoch, unit="s", tz="UTC")).total_seconds()), 1e-6)

    def test_epoch_reset_and_uint32_max_remain_parseable(self):
        decoded, units = decode_unix(pd.Series(["0", "4294967295", "invalid"]))
        self.assertEqual(units.tolist(), ["s", "s", "unrecognized"])
        self.assertEqual(decoded.iloc[0], pd.Timestamp("1970-01-01", tz="UTC"))
        self.assertEqual(decoded.iloc[1], pd.Timestamp("2106-02-07 06:28:15", tz="UTC"))
        self.assertTrue(pd.isna(decoded.iloc[2]))

    def test_agreeing_fields_can_both_contain_future_clock_error(self):
        raw = pd.DataFrame({"id": [1, 2], "device_id": ["a", "a"],
                            "date_time": ["2026-04-13 10:23:55", "2106-02-07 06:28:15"],
                            "unixtime": ["1776075835", "4294967295"],
                            "created_at": ["2026-04-13 17:24:55", "2026-04-13 17:25:55"],
                            **{sensor: [1.0, 1.0] for sensor in FEATURES}})
        with tempfile.TemporaryDirectory() as folder:
            _, summary = audit(raw, folder, pd.Timestamp("2026-11-15"))
        self.assertEqual(summary["invalid_unixtime"], 0)
        self.assertEqual(summary["date_time_future_rows"], 1)
        self.assertEqual(summary["timezone_checks"]["UTC"]["agree_within_2s"], 2)
        self.assertEqual(summary["timezone_checks"]["Asia/Jakarta"]["agree_within_2s"], 0)


if __name__ == "__main__":
    unittest.main()
