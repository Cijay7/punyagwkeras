"""Kontrak REST dan pengolahan input; tidak bergantung pada proyek di luar folder."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

FEATURES = ["SM", "ST", "ph", "SEC", "WL", "RR"]
LAGS, HORIZON = 24, 24
DEVICE_PATTERN = r"^[A-Za-z0-9_-]{1,100}$"
DeviceName = Annotated[str, Field(pattern=DEVICE_PATTERN)]
Mode = Literal["demo", "live"]


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date_time: AwareDatetime
    SM: float | None
    ST: float | None
    ph: float | None
    SEC: float | None
    WL: float | None
    RR: float | None


class History(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device_id: DeviceName
    time_is_synthetic: bool
    readings: list[Reading] = Field(min_length=1, max_length=2000)


class DeviceSummary(BaseModel):
    device_id: DeviceName
    latest_time: AwareDatetime


class Devices(BaseModel):
    devices: list[DeviceSummary] = Field(max_length=2000)


class Unavailable(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


def prepare(history, timezone, now):
    rows = [{"date_time": pd.Timestamp(r.date_time).tz_convert(timezone).tz_localize(None),
             **{s: getattr(r, s) for s in FEATURES}} for r in history.readings]
    raw = pd.DataFrame(rows).sort_values("date_time", kind="stable").drop_duplicates("date_time", keep="last")
    raw[FEATURES] = raw[FEATURES].astype("float64")
    if raw.date_time.max() > now:
        raise Unavailable("future_reading", "Input memuat pembacaan dari masa depan.")
    last = raw.iloc[-1]
    origin = pd.Timestamp(last.date_time).floor("h")
    if origin < pd.Timestamp("2000-01-01"):
        raise Unavailable("invalid_time", "Jam input terakhir tidak wajar untuk model ini.")
    if last[["SM", "ST", "ph", "SEC"]].eq(0).all():
        raise Unavailable("invalid_latest_reading", "Paket sensor tanah terakhir nol bersamaan.")
    suspect = raw[["SM", "ST", "ph", "SEC"]].eq(0).all(axis=1)
    raw.loc[suspect, FEATURES] = np.nan
    # Cukup rentang input terakhir; timestamp lama tidak membuat resample puluhan tahun.
    raw = raw.loc[raw.date_time >= origin-pd.Timedelta(hours=LAGS)]
    hourly = raw.set_index("date_time")[FEATURES].resample("1h", closed="right", label="right").mean()
    required = pd.date_range(origin-pd.Timedelta(hours=LAGS-1), periods=LAGS, freq="h")
    window = hourly.reindex(required)
    if window.isna().any(axis=None) or not np.isfinite(window.to_numpy()).all():
        raise Unavailable("insufficient_history", "Perlu 24 jam input terbaru lengkap; data 24 baris belum tentu mencakup 24 jam.")
    return window, origin


def sequence(window, sensor, scale):
    # 24 nilai terlama -> terbaru; sama dengan training Keras asli.
    values = window[sensor].to_numpy(dtype=np.float64)
    level = values[-1]
    hours = window.index.hour.to_numpy()
    inputs = np.stack([(values-level)/scale, np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)], axis=-1)
    return inputs.astype("float32")[None, :, :], level


def bounded(values, sensor):
    if sensor == "SM":
        return np.clip(values, 0, 100)
    if sensor == "ph":
        return np.clip(values, 0, 14)
    if sensor in ("SEC", "RR"):
        return np.maximum(values, 0)
    return values


def validate_rules(rules):
    if not isinstance(rules, dict) or not isinstance(rules.get("enabled"), bool):
        raise ValueError("Aturan harus mempunyai enabled boolean.")
    if not rules["enabled"]:
        return
    if not rules.get("source") or not rules.get("units_and_reference_confirmed") or not rules.get("rules"):
        raise ValueError("Aturan aktif memerlukan sumber, satuan dan daftar ambang.")
    for rule in rules["rules"]:
        if rule.get("sensor") not in FEATURES or rule.get("direction") not in ("low", "high"):
            raise ValueError("Sensor/arah aturan tidak sesuai.")
        a, b = rule.get("rawan"), rule.get("bahaya")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not np.isfinite([a, b]).all():
            raise ValueError("Ambang harus finite.")
        if not (b < a if rule["direction"] == "low" else b > a):
            raise ValueError("Urutan ambang tidak sesuai.")


def classify(values, rules, stale):
    if stale:
        return {"status": "Data lama", "alasan": "Input historis; bukan peringatan kondisi saat ini"}
    if not rules["enabled"]:
        return {"status": "Belum dikonfigurasi", "alasan": "Belum ada ambang terverifikasi yang aktif"}
    level, reasons = 0, []
    labels = ["Normal", "Rawan", "Bahaya"]
    for rule in rules["rules"]:
        value = values[rule["sensor"]]
        if rule["direction"] == "low":
            score = 2 if value <= rule["bahaya"] else 1 if value <= rule["rawan"] else 0
        else:
            score = 2 if value >= rule["bahaya"] else 1 if value >= rule["rawan"] else 0
        level = max(level, score)
        if score:
            reasons.append(f"{rule['sensor']}={value:.3g}: {labels[score]}")
    return {"status": labels[level], "alasan": "; ".join(reasons) or "Tidak ada ambang yang terlampaui"}
