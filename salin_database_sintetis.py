"""Salin sensor_data ke database baru dengan waktu sintetis per device."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

EXTRA_COLUMNS = ["device_sequence", "original_id", "original_date_time", "original_unixtime", "time_is_synthetic"]


def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value or ""):
        raise ValueError("Nama database/tabel harus berupa identifier sederhana, maksimal 64 karakter.")
    return f"`{value}`"


def reconstruct(source, start="2026-01-01 00:00:00", timezone="Asia/Jakarta", interval_seconds=600):
    """ID global dari 0; waktu mengikuti cumcount per device pada urutan ID asli."""
    if source.empty:
        raise ValueError("Sumber kosong; tidak membuat database baru.")
    if source.id.isna().any() or source.id.duplicated().any():
        raise ValueError("ID asli harus unik dan tidak kosong.")
    if source.device_id.isna().any() or source.device_id.astype(str).str.strip().eq("").any():
        raise ValueError("Semua baris harus mempunyai device_id.")
    if set(EXTRA_COLUMNS).intersection(source.columns):
        raise ValueError("Sumber sudah memiliki kolom rekonstruksi; hindari penyalinan sintetis berulang.")
    if interval_seconds <= 0:
        raise ValueError("Interval harus positif.")
    start_time = pd.Timestamp(start)
    start_time = start_time.tz_localize(timezone) if start_time.tzinfo is None else start_time.tz_convert(timezone)
    if start_time.microsecond or start_time.nanosecond:
        raise ValueError("Waktu awal harus mempunyai presisi detik bulat.")
    result = source.sort_values("id", kind="stable").reset_index(drop=True).copy()
    result["original_id"] = result.id.astype("int64")
    result["original_date_time"] = result.date_time
    result["original_unixtime"] = result.unixtime
    result["device_sequence"] = result.groupby("device_id", sort=False).cumcount().astype("int64")
    epoch = int(start_time.timestamp()) + result.device_sequence.to_numpy() * interval_seconds
    result["id"] = np.arange(len(result), dtype=np.int64)
    result["unixtime"] = epoch.astype("int64")
    result["date_time"] = pd.to_datetime(epoch, unit="s", utc=True).tz_convert(timezone).tz_localize(None)
    result["time_is_synthetic"] = 1
    return result


def verify_copy(source, copied, timezone, interval_seconds):
    source = source.sort_values("id", kind="stable").reset_index(drop=True)
    copied = copied.sort_values("original_id", kind="stable").reset_index(drop=True)
    if len(source) != len(copied):
        raise ValueError("Jumlah baris salinan tidak sama.")
    np.testing.assert_array_equal(copied.id.to_numpy(), np.arange(len(source)))
    np.testing.assert_array_equal(copied.original_id.to_numpy(), source.id.to_numpy())
    np.testing.assert_array_equal(copied.device_sequence.to_numpy(), copied.groupby("device_id", sort=False).cumcount().to_numpy())
    if not copied.time_is_synthetic.eq(1).all():
        raise ValueError("Penanda waktu sintetis harus aktif untuk semua baris.")
    # Semua kolom sumber dipertahankan; hanya tiga kolom ini dialihkan ke arsip.
    restore = {"id": "original_id", "date_time": "original_date_time", "unixtime": "original_unixtime"}
    for column in source.columns:
        pd.testing.assert_series_equal(source[column], copied[restore.get(column, column)],
                                       check_dtype=False, check_names=False, check_exact=True)
    converted = pd.to_datetime(copied.unixtime, unit="s", utc=True).dt.tz_convert(timezone).dt.tz_localize(None)
    pd.testing.assert_series_equal(pd.to_datetime(copied.date_time), converted, check_names=False)
    for _, part in copied.groupby("device_id", sort=False):
        if not part.unixtime.diff().iloc[1:].eq(interval_seconds).all():
            raise ValueError("Interval unixtime per device tidak sesuai.")
        if not pd.to_datetime(part.date_time).diff().iloc[1:].eq(pd.Timedelta(seconds=interval_seconds)).all():
            raise ValueError("Interval date_time per device tidak sesuai.")
    first = copied.groupby("device_id", sort=False).first()
    if first.date_time.nunique() != 1:
        raise ValueError("Semua device harus mulai dari waktu yang sama.")
    return {"rows": len(copied), "devices": int(copied.device_id.nunique()),
            "id_min": int(copied.id.min()), "id_max": int(copied.id.max()),
            "all_sensor_and_original_fields_preserved": True,
            "all_device_sequences_start_at_zero": True,
            "all_intervals_seconds": interval_seconds, "date_time_unixtime_consistent": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="tugasakhir_sintetis_10menit")
    parser.add_argument("--start", default="2026-01-01 00:00:00")
    parser.add_argument("--timezone", default="Asia/Jakarta")
    parser.add_argument("--interval-minutes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("hasil_salin_sintetis"))
    parser.add_argument("--write", action="store_true", help="Buat database baru dan tulis salinan")
    args = parser.parse_args()
    url = make_url(os.getenv("MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir"))
    source_name = url.database
    source_db, target_db = identifier(source_name), identifier(args.target)
    if source_name.lower() == args.target.lower():
        parser.error("Target harus database baru yang berbeda dari sumber.")
    # created_at bertipe TIMESTAMP dibaca dan ditulis pada session UTC yang sama.
    engine = create_engine(url, connect_args={"connect_timeout": 10, "init_command": "SET time_zone = '+00:00'"}, hide_parameters=True)
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
            with conn.begin():
                source = pd.read_sql(text(f"SELECT * FROM {source_db}.sensor_data ORDER BY id"), conn)
        copied = reconstruct(source, args.start, args.timezone, args.interval_minutes * 60)
        checks = verify_copy(source, copied, args.timezone, args.interval_minutes * 60)
        args.output.mkdir(parents=True, exist_ok=True)
        summary = copied.groupby("device_id", sort=True).agg(rows=("id", "size"),
                    sequence_start=("device_sequence", "min"), sequence_end=("device_sequence", "max"),
                    date_time_start=("date_time", "min"), date_time_end=("date_time", "max"))
        summary.to_csv(args.output / "ringkasan_per_device.csv")
        preview = pd.concat([copied.groupby("device_id", sort=True).head(3), copied.groupby("device_id", sort=True).tail(1)])
        preview.sort_values(["device_id", "device_sequence"])[["id", "device_id", "device_sequence", "date_time", "unixtime", "original_id", "original_date_time"]].to_csv(args.output / "preview.csv", index=False)
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(source, index=True).values.tobytes()).hexdigest()
        metadata = {"source_database": source_name, "target_database": args.target, "source_table": "sensor_data",
                    "start_time_local": str(copied.date_time.iloc[0]), "timezone": args.timezone,
                    "interval_seconds": args.interval_minutes * 60, "source_fingerprint": fingerprint,
                    "time_semantics": "synthetic_not_actual_measurement",
                    "id_strategy": "global_id_from_zero_and_device_sequence_from_zero",
                    "status": "preview", "verification": checks}
        if args.write:
            with engine.connect() as conn:
                exists = conn.execute(text("SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name=:name"), {"name": args.target}).scalar_one()
                if exists:
                    raise ValueError("Database target sudah ada; tidak ditimpa. Pilih nama database baru.")
                conn.execute(text(f"CREATE DATABASE {target_db} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.execute(text(f"CREATE TABLE {target_db}.sensor_data LIKE {source_db}.sensor_data"))
                conn.execute(text(f"""ALTER TABLE {target_db}.sensor_data
                    MODIFY COLUMN id BIGINT NOT NULL,
                    ADD COLUMN device_sequence BIGINT NOT NULL,
                    ADD COLUMN original_id BIGINT NOT NULL,
                    ADD COLUMN original_date_time DATETIME NULL,
                    ADD COLUMN original_unixtime BIGINT NULL,
                    ADD COLUMN time_is_synthetic TINYINT NOT NULL DEFAULT 1,
                    ADD UNIQUE KEY uq_original_id (original_id),
                    ADD UNIQUE KEY uq_device_sequence (device_id, device_sequence),
                    ADD KEY idx_device_time (device_id, date_time, id)"""))
                conn.execute(text(f"""CREATE TABLE {target_db}.dataset_metadata (
                    id INT NOT NULL PRIMARY KEY,
                    details JSON NOT NULL
                ) ENGINE=InnoDB"""))
                conn.commit()
                print("Database baru dibuat; menulis salinan...", flush=True)
                with conn.begin():
                    copied.to_sql("sensor_data", conn, schema=args.target, if_exists="append", index=False, chunksize=2000)
                    readback = pd.read_sql(text(f"SELECT * FROM {target_db}.sensor_data ORDER BY original_id"), conn)
                    checks = verify_copy(source, readback, args.timezone, args.interval_minutes * 60)
                    metadata.update(status="complete", verification=checks)
                    conn.execute(text(f"INSERT INTO {target_db}.dataset_metadata (id,details) VALUES (1,:details)"),
                                 {"details": json.dumps(metadata)})
                # Pastikan hasil benar-benar terlihat setelah commit.
                committed = conn.execute(text(f"SELECT COUNT(*) FROM {target_db}.sensor_data")).scalar_one()
                if committed != len(source):
                    raise ValueError("Jumlah baris setelah commit tidak sesuai.")
                # Baca ulang sumber untuk memastikan tidak ada perubahan akibat proses salin.
                after = pd.read_sql(text(f"SELECT * FROM {source_db}.sensor_data ORDER BY id"), conn)
                after_hash = hashlib.sha256(pd.util.hash_pandas_object(after, index=True).values.tobytes()).hexdigest()
                metadata["source_unchanged_at_verification"] = after_hash == fingerprint
        (args.output / "hasil.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        print(summary.to_string())
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
