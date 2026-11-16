"""Audit read-only date_time vs unixtime dan dampak alternatif waktu pada training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from ews_gambut import FEATURES, HORIZON, LAGS, prepare_device

QUERY = """
SELECT id, device_id, CAST(date_time AS CHAR) AS date_time,
       CAST(unixtime AS CHAR) AS unixtime, CAST(created_at AS CHAR) AS created_at,
       soil_moisture AS SM, soil_temperature AS ST, ph_value AS ph,
       soil_conductivity AS SEC, water_level AS WL, rain_meter AS RR
FROM sensor_data ORDER BY device_id, id
"""


def decode_unix(series):
    numeric = pd.to_numeric(series, errors="coerce")
    units = pd.Series("unrecognized", index=series.index)
    result = pd.Series(pd.NaT, dtype="datetime64[ns, UTC]", index=series.index)
    # Inferensi magnitudo tidak membuang nilai epoch 1970/2106: keduanya tetap
    # dapat dikonversi walaupun tidak wajar sebagai waktu pengukuran proyek ini.
    for unit, low, high in (("s", 0, 1e11), ("ms", 1e11, 1e14),
                            ("us", 1e14, 1e17), ("ns", 1e17, 1e20)):
        mask = numeric.abs().ge(low) & numeric.abs().lt(high)
        result.loc[mask] = pd.to_datetime(numeric.loc[mask], unit=unit, utc=True, errors="coerce")
        units.loc[mask] = unit
    return result, units


def audit(raw, output, as_of):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = raw.copy()
    dt = pd.to_datetime(frame.date_time, errors="coerce")
    created = pd.to_datetime(frame.created_at, errors="coerce")
    unix, units = decode_unix(frame.unixtime)
    frame["parsed_date_time"], frame["parsed_created_at"] = dt, created
    frame["unix_unit"], frame["unix_utc"] = units, unix
    summary = {"total_rows": len(frame), "devices": int(frame.device_id.nunique()), "as_of_jakarta": str(as_of),
               "invalid_date_time": int(dt.isna().sum()), "invalid_unixtime": int(unix.isna().sum()),
               "unixtime_units": units.value_counts().to_dict(),
               "date_time_min": str(dt.min()), "date_time_max": str(dt.max()),
               "unix_utc_min": str(unix.min()), "unix_utc_max": str(unix.max()),
               "date_time_before_2026_rows": int(dt.lt(pd.Timestamp("2026-01-01")).sum()),
               "unixtime_outside_2000_2100": int((unix.lt(pd.Timestamp("2000-01-01", tz="UTC")) |
                                                unix.ge(pd.Timestamp("2100-01-01", tz="UTC"))).sum()),
               "date_time_future_rows": int(dt.gt(as_of).sum()), "timezone_checks": {}}
    policies = {"date_time": dt}
    for timezone in ("UTC", "Asia/Jakarta", "Asia/Makassar"):
        converted = unix.dt.tz_convert(timezone).dt.tz_localize(None)
        name = timezone.replace("/", "_")
        frame[f"unix_{name}"] = converted
        delta = (dt - converted).dt.total_seconds()
        frame[f"delta_{name}_seconds"] = delta
        summary["timezone_checks"][timezone] = {
            "valid_pairs": int(delta.notna().sum()), "agree_within_2s": int(delta.abs().le(2).sum()),
            "different_over_1h": int(delta.abs().gt(3600).sum()),
            "median_delta_seconds": float(delta.median()) if delta.notna().any() else None,
            "common_offsets_seconds": {str(k): int(v) for k, v in delta.value_counts().head(8).items()}}
        policies[f"unixtime_{name}"] = converted
    frame["date_minus_created_seconds"] = (dt - created).dt.total_seconds()
    frame["receipt_delay_if_date_UTC_seconds"] = (created - (dt + pd.Timedelta(hours=7))).dt.total_seconds()
    delay = frame.receipt_delay_if_date_UTC_seconds
    summary["receipt_delay_assuming_created_jakarta_date_utc"] = {
        "median_seconds": float(delay.median()), "within_0_300s": int(delay.between(0, 300).sum()),
        "outside_plus_minus_1day": int(delay.abs().gt(86400).sum())}
    used = dt.ge(pd.Timestamp("2026-01-01")) & dt.le(as_of)
    summary["rows_passing_existing_training_time_filter"] = int(used.sum())
    summary["passing_filter_but_receipt_gap_over_1day"] = int((used & delay.abs().gt(86400)).sum())
    summary["common_date_minus_created_seconds"] = {str(k): int(v) for k, v in frame.date_minus_created_seconds.round().value_counts().head(10).items()}
    device_rows, impact_rows = [], []
    for device, part in frame.groupby("device_id", sort=True):
        ordered = part.sort_values("id")
        pdt = ordered.parsed_date_time
        punix = ordered.unix_utc
        delta = ordered.delta_UTC_seconds
        device_rows.append({"device_id": str(device), "rows": len(part),
                            "utc_agree_2s": int(delta.abs().le(2).sum()),
                            "utc_disagree_over_1h": int(delta.abs().gt(3600).sum()),
                            "date_time_min": str(pdt.min()), "date_time_max": str(pdt.max()),
                            "unix_utc_min": str(punix.min()), "unix_utc_max": str(punix.max()),
                            "date_time_backwards_by_id": int(pdt.diff().dt.total_seconds().lt(0).sum()),
                            "unix_backwards_by_id": int(punix.diff().dt.total_seconds().lt(0).sum()),
                            "date_duplicate_rows": int(pdt.duplicated().sum()),
                            "unix_duplicate_rows": int(punix.duplicated().sum()),
                            "date_time_future": int(pdt.gt(as_of).sum()),
                            "rows_passing_time_filter": int(used.loc[part.index].sum()),
                            "passing_filter_receipt_gap_over_1day": int((used.loc[part.index] & part.receipt_delay_if_date_UTC_seconds.abs().gt(86400)).sum())})
    for policy, times in policies.items():
        eligible = times.ge(pd.Timestamp("2026-01-01")) & times.le(as_of)
        alternate = raw.loc[eligible].copy()
        alternate["date_time"] = times.loc[eligible]
        for device in raw.device_id.dropna().unique():
            part = alternate.loc[alternate.device_id.eq(device)]
            if part.empty:
                continue
            hourly, prep = prepare_device(part, device)
            row = {"policy": policy, "device_id": str(device), "input_rows": len(part),
                   "hourly_rows": len(hourly), "complete_hours": prep["complete_hours"],
                   "suspect_zero_packets": prep["suspect_zero_packets"]}
            for sensor in FEATURES:
                row[f"windows_{sensor}"] = int(hourly[sensor].notna().rolling(LAGS + HORIZON).sum().eq(LAGS + HORIZON).sum())
            impact_rows.append(row)
    anomaly = (dt.isna() | unix.isna() | frame.delta_UTC_seconds.abs().gt(2) | dt.gt(as_of))
    columns = ["id", "device_id", "date_time", "unixtime", "created_at", "unix_unit", "unix_utc", "delta_UTC_seconds", "delta_Asia_Jakarta_seconds"]
    # Pemilihan contoh deterministik; hitungan penuh ada di summary/per_device.
    frame.loc[anomaly, columns].groupby("device_id", group_keys=False).head(20).to_csv(output / "contoh_anomali.csv", index=False)
    frame[columns].groupby("device_id", group_keys=False).tail(3).to_csv(output / "contoh_terakhir_per_device.csv", index=False)
    pd.DataFrame(device_rows).to_csv(output / "audit_per_device.csv", index=False)
    impact = pd.DataFrame(impact_rows)
    impact.to_csv(output / "dampak_jendela_training.csv", index=False)
    summary["training_windows_total_by_policy"] = impact.groupby("policy")[[f"windows_{s}" for s in FEATURES]].sum().to_dict("index")
    (output / "ringkasan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("audit_timestamp"))
    args = parser.parse_args()
    engine = create_engine(os.getenv("MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir"),
                           connect_args={"connect_timeout": 5}, hide_parameters=True)
    try:
        with engine.connect() as conn:
            raw = pd.read_sql(text(QUERY), conn)
            server = pd.read_sql(text("SELECT @@session.time_zone AS session_timezone, @@system_time_zone AS system_timezone, NOW() AS server_now, UTC_TIMESTAMP() AS server_utc"), conn)
    finally:
        engine.dispose()
    as_of = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    _, summary = audit(raw, args.output, as_of)
    server.to_csv(args.output / "waktu_server.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
