"""Training per parameter lintas perangkat; prediksi sensor 24 jam per perangkat. Tidak menulis ke MySQL."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from parameter_models import select_parameter_model

FEATURES = ["SM", "ST", "ph", "SEC", "WL", "RR"]
COLUMNS = {
    "soil_moisture": "SM", "soil_temperature": "ST", "ph_value": "ph",
    "soil_conductivity": "SEC", "water_level": "WL", "rain_meter": "RR",
}
HORIZON = 24
LAGS = 24
LABELS = ["Normal", "Rawan", "Bahaya"]


def load_mysql(start="2026-01-01", as_of=None):
    # date_time dipertahankan sebagaimana disimpan; zona waktu database perlu dikonfirmasi.
    as_of = pd.Timestamp(as_of or pd.Timestamp.now(tz="Asia/Makassar").tz_localize(None))
    engine = create_engine(os.getenv(
        "MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir"
    ), connect_args={"connect_timeout": 10}, hide_parameters=True)
    query = """
        SELECT id, device_id, date_time,
               soil_moisture AS SM, soil_temperature AS ST, ph_value AS ph,
               soil_conductivity AS SEC, water_level AS WL, rain_meter AS RR
        FROM sensor_data
        WHERE date_time >= :start AND date_time <= :as_of
        ORDER BY device_id, date_time, id
    """
    try:
        with engine.connect() as conn:
            raw = pd.read_sql(text(query), conn, params={
                "start": str(pd.Timestamp(start)), "as_of": str(as_of),
            })
            coverage = pd.read_sql(text("""
                SELECT COUNT(*) AS total_rows,
                       SUM(date_time IS NULL) AS null_time,
                       SUM(date_time < :start) AS before_start,
                       SUM(date_time > :as_of) AS future_time
                FROM sensor_data
            """), conn, params={"start": str(pd.Timestamp(start)), "as_of": str(as_of)})
    finally:
        engine.dispose()
    return raw, coverage


def prepare_device(raw, device_id, exclude_zero_packet=True):
    data = raw.loc[raw.device_id.eq(device_id)].copy()
    data["date_time"] = pd.to_datetime(data.date_time, errors="coerce")
    invalid_times = int(data.date_time.isna().sum())
    data = data.dropna(subset=["date_time"]).sort_values(["date_time", "id"])
    duplicates = int(data.duplicated("date_time", keep="last").sum())
    data = data.drop_duplicates("date_time", keep="last")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Paket empat sensor tanah bernilai nol bersamaan ditandai mencurigakan.
    # Kebijakan ini eksplisit dan bisa dinonaktifkan setelah pemeriksaan alat.
    suspect = data[["SM", "ST", "ph", "SEC"]].eq(0).all(axis=1)
    suspect_count = int(suspect.sum())
    if exclude_zero_packet:
        data.loc[suspect, FEATURES] = np.nan
    data = data.set_index("date_time")
    # Bin berlabel waktu akhir: jam t hanya berisi pengukuran <= t.
    # RR memakai rata-rata pembacaan, BUKAN akumulasi hujan, hingga definisinya pasti.
    hourly = data[FEATURES].resample("1h", closed="right", label="right").mean()
    if len(data):
        hourly = hourly.loc[hourly.index <= data.index.max().floor("h")]
    audit = {"device_id": device_id, "rows": len(data), "duplicate_times": duplicates,
             "invalid_times": invalid_times, "suspect_zero_packets": suspect_count,
             "hourly_rows": len(hourly), "complete_hours": int(hourly.notna().all(axis=1).sum()),
             "latest_measurement": str(data.index.max())}
    audit["constant_sensors"] = ", ".join(sensor for sensor in FEATURES if hourly[sensor].nunique() <= 1)
    # Celah tetap NaN: tidak mengisi data dengan nilai dari masa depan.
    return hourly, audit


def predictors(hourly):
    parts = [hourly.shift(lag).add_suffix(f"_lag{lag}") for lag in range(LAGS)]
    result = pd.concat(parts, axis=1)
    result["hour_sin"] = np.sin(2 * np.pi * result.index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result.index.hour / 24)
    return result


def supervised(hourly):
    x = predictors(hourly)
    y = pd.concat([hourly.shift(-h).add_suffix(f"_h{h}") for h in range(1, HORIZON + 1)], axis=1)
    valid = x.notna().all(axis=1) & y.notna().all(axis=1)
    return x.loc[valid], y.loc[valid]


def physical_bounds(values):
    shaped = np.asarray(values).reshape(-1, HORIZON, len(FEATURES)).copy()
    shaped[:, :, 0] = np.clip(shaped[:, :, 0], 0, 100)
    shaped[:, :, 2] = np.clip(shaped[:, :, 2], 0, 14)
    shaped[:, :, 3] = np.maximum(shaped[:, :, 3], 0)
    shaped[:, :, 5] = np.maximum(shaped[:, :, 5], 0)
    # WL tidak dibatasi karena arah/nol referensinya belum diketahui.
    return shaped


def parameter_bounds(values, sensor):
    values = np.asarray(values).copy()
    if sensor == "SM":
        return np.clip(values, 0, 100)
    if sensor == "ph":
        return np.clip(values, 0, 14)
    if sensor in ("SEC", "RR"):
        return np.maximum(values, 0)
    return values


def train_parameters(hourly_by_device, strategy="conservative"):
    """Satu model per parameter; lag/target dibuat sebelum data perangkat digabung."""
    if strategy not in ("improved", "legacy", "conservative"):
        raise ValueError("Strategi training tidak dikenal.")
    bundles, metrics = {}, []
    for sensor in FEATURES:
        samples = {}
        for device, hourly in hourly_by_device.items():
            x, y = supervised(hourly[[sensor]])
            if len(x):
                samples[device] = (x, y)
        if not samples:
            raise ValueError(f"{sensor}: tidak ada jendela training lengkap.")
        x = pd.concat({device: pair[0] for device, pair in samples.items()}, names=["device_id", "date_time"])
        y = pd.concat({device: pair[1] for device, pair in samples.items()}, names=["device_id", "date_time"])
        times = x.index.get_level_values("date_time")
        origins = times.unique().sort_values()
        if len(origins) < 168:
            raise ValueError(f"{sensor}: hanya {len(origins)} waktu asal lengkap; perlu minimal 168.")
        # Batas waktu global mencegah data masa depan perangkat lain masuk train.
        validation_time = origins[int(len(origins) * 0.6)]
        test_time = origins[int(len(origins) * 0.8)]
        train = times + pd.Timedelta(hours=HORIZON) < validation_time
        validation = (times >= validation_time) & (times + pd.Timedelta(hours=HORIZON) < test_time)
        test = times >= test_time
        if min(train.sum(), validation.sum(), test.sum()) < 24:
            raise ValueError(f"{sensor}: blok train/validation/test terlalu pendek.")
        validation_scores = []
        # Kebijakan rilis dari benchmark dua periode; bukan estimasi test independen baru.
        use_candidates = strategy == "improved" or (strategy == "conservative" and sensor in ("SM", "ST"))
        if use_candidates:
            model, validation_scores = select_parameter_model(
                x.loc[train], y.loc[train], x.loc[validation], y.loc[validation], sensor)
            method = "Seleksi per horizon"
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
            model.fit(x.loc[train], y.loc[train])
            actual = y.loc[validation].to_numpy()
            predicted = parameter_bounds(model.predict(x.loc[validation]), sensor)
            baseline = x.loc[validation, f"{sensor}_lag0"].to_numpy()[:, None]
            use_ridge = (strategy == "legacy" and
                         np.abs(predicted - actual).mean() < np.abs(baseline - actual).mean())
            method = "Ridge" if use_ridge else "Nilai terakhir"
        before_test = times + pd.Timedelta(hours=HORIZON) < test_time
        model.fit(x.loc[before_test], y.loc[before_test])
        actual = y.loc[test].to_numpy()
        baseline = np.repeat(x.loc[test, f"{sensor}_lag0"].to_numpy()[:, None], HORIZON, axis=1)
        chosen = baseline if method == "Nilai terakhir" else parameter_bounds(model.predict(x.loc[test]), sensor)
        varying = np.ptp(x.loc[test, [f"{sensor}_lag{i}" for i in range(LAGS)]].to_numpy(), axis=1) > 0
        device_metrics = []
        test_devices = x.index.get_level_values("device_id")[test]
        for device in test_devices.unique():
            mask = test_devices == device
            device_metrics.append({"device_id": str(device), "sensor": sensor,
                                   "test_origins": int(mask.sum()),
                                   "nonconstant_input_origins": int((varying & mask).sum()),
                                   "mae": float(np.abs(chosen[mask] - actual[mask]).mean()),
                                   "mae_baseline": float(np.abs(baseline[mask] - actual[mask]).mean())})
        for h in range(HORIZON):
            metrics.append({"horizon_jam": h + 1, "sensor": sensor, "metode": model.methods[h] if use_candidates else method,
                            "mae": float(np.abs(chosen[:, h] - actual[:, h]).mean()),
                            "mae_baseline": float(np.abs(baseline[:, h] - actual[:, h]).mean()),
                            "test_origins": int(test.sum()),
                            "nonconstant_input_origins": int(varying.sum()),
                            "mae_nonconstant_input": float(np.abs(chosen[varying, h] - actual[varying, h]).mean()) if varying.any() else None,
                            "mae_baseline_nonconstant_input": float(np.abs(baseline[varying, h] - actual[varying, h]).mean()) if varying.any() else None})
        warning = (f"MAE test melebihi baseline: {sensor}" if
                   np.abs(chosen - actual).mean() > np.abs(baseline - actual).mean() else
                   "Periksa MAE per horizon; belum memvalidasi risiko kebakaran")
        model.fit(x, y)
        bundles[sensor] = {"model": model, "metadata": {
            "training_scope": "parameter", "sensor": sensor,
            "training_devices": [str(device) for device in samples],
            "trained_until": str((times + pd.Timedelta(hours=HORIZON)).max()),
            "train_windows": int(train.sum()), "validation_windows": int(validation.sum()),
            "test_windows": int(test.sum()), "validation_start": str(validation_time),
            "test_start": str(test_time), "lag_hours": LAGS, "horizon_hours": HORIZON,
            "method": method, "evaluation_warning": warning,
            "strategy": strategy, "validation_scores": validation_scores,
            "method_per_horizon": model.methods if use_candidates else [method] * HORIZON,
            "test_per_device": device_metrics,
            "test_nonconstant_input_origins": int(varying.sum()),
            "release_policy": "SM/ST kandidat; parameter lain persistence. Kebijakan dari benchmark historis; perlu data baru."}}
    return bundles, pd.DataFrame(metrics)


def predict_parameters(hourly, bundles):
    """Gunakan keenam model bersama untuk histori satu perangkat, tanpa fit."""
    if hourly.empty:
        raise ValueError("Tidak ada data per jam.")
    future = np.empty((HORIZON, len(FEATURES)))
    for j, sensor in enumerate(FEATURES):
        last_x = predictors(hourly[[sensor]]).iloc[[-1]]
        if last_x.isna().any(axis=None):
            raise ValueError(f"24 jam input terakhir {sensor} tidak lengkap.")
        bundle = bundles[sensor]
        if pd.Timestamp(bundle["metadata"]["trained_until"]) > hourly.index[-1]:
            raise ValueError("Model dilatih melewati waktu input perangkat.")
        if bundle["metadata"]["method"] == "Nilai terakhir":
            future[:, j] = hourly.iloc[-1][sensor]
        else:
            values = np.asarray(bundle["model"].predict(last_x))
            if values.shape != (1, HORIZON) or not np.isfinite(values).all():
                raise ValueError(f"Keluaran model {sensor} tidak valid.")
            future[:, j] = parameter_bounds(values[0], sensor)
    return future


def forecast_parameters(hourly, bundles, as_of=None):
    future = predict_parameters(hourly, bundles)
    origin = hourly.index[-1]
    result = pd.DataFrame(future, columns=FEATURES,
                          index=pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON, freq="h"))
    result.index.name = "date_time"
    as_of = pd.Timestamp(as_of or pd.Timestamp.now(tz="Asia/Makassar").tz_localize(None))
    stale = as_of - origin > pd.Timedelta(hours=2)
    result["forecast_origin"] = origin
    result["data_lama"] = bool(stale)
    result["jenis"] = "Prediksi historis; bukan kondisi saat ini" if stale else "Prediksi 24 jam"
    result["catatan_evaluasi"] = "; ".join(dict.fromkeys(
        bundle["metadata"]["evaluation_warning"] for bundle in bundles.values()))
    return result


def validate_rules(config):
    if not config.get("enabled"):
        return
    if not config.get("source") or not config.get("units_and_reference_confirmed"):
        raise ValueError("Aturan perlu sumber serta konfirmasi satuan dan referensi pengukuran.")
    if not config.get("rules"):
        raise ValueError("Aturan aktif tetapi daftar rules kosong.")
    for rule in config["rules"]:
        if rule.get("sensor") not in FEATURES or rule.get("direction") not in ("low", "high"):
            raise ValueError("Sensor atau arah aturan tidak valid.")
        a, b = rule.get("rawan"), rule.get("bahaya")
        if a is None or b is None or not np.isfinite([a, b]).all():
            raise ValueError("Ambang Rawan dan Bahaya harus berupa angka terverifikasi.")
        if not (b < a if rule["direction"] == "low" else b > a):
            raise ValueError("Urutan ambang Rawan/Bahaya salah.")


def classify_rules(frame, config):
    """Skor ordinal = maksimum tingkat pemicu. Bukan probabilitas kebakaran."""
    validate_rules(config)
    result = pd.DataFrame(index=frame.index)
    result["status"] = "Belum dikonfigurasi"
    result["alasan"] = "Belum ada label atau aturan ambang terverifikasi"
    if config.get("enabled"):
        for idx, row in frame.iterrows():
            required = [rule["sensor"] for rule in config["rules"]]
            if not np.isfinite(pd.to_numeric(row[required], errors="coerce")).all():
                result.loc[idx] = ["Data tidak lengkap", "Sensor untuk aturan tidak tersedia"]
                continue
            level, reasons = 0, []
            for rule in config["rules"]:
                value = row[rule["sensor"]]
                if rule["direction"] == "low":
                    score = 2 if value <= rule["bahaya"] else 1 if value <= rule["rawan"] else 0
                else:
                    score = 2 if value >= rule["bahaya"] else 1 if value >= rule["rawan"] else 0
                level = max(level, score)
                if score:
                    reasons.append(f"{rule['sensor']}={value:.3g}: {LABELS[score]}")
            result.loc[idx] = [LABELS[level], "; ".join(reasons) or "Tidak ada ambang aturan yang terlampaui"]
    # Data historis tidak boleh menghasilkan status peringatan operasional saat ini.
    if "data_lama" in frame:
        result.loc[frame.data_lama, "status"] = "Data lama"
        result.loc[frame.data_lama, "alasan"] = "Perbarui data sensor sebelum menentukan peringatan saat ini"
    return result


def run_all(raw, config, output_dir="hasil_ews", as_of=None):
    import joblib
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_rules(config)
    forecasts, audits, hourly_by_device = [], [], {}
    for device in sorted(raw.device_id.dropna().unique()):
        hourly, audit = prepare_device(raw, device)
        hourly_by_device[device] = hourly
        audits.append(audit)
    # Enam model parameter dilatih di luar loop inferensi perangkat.
    bundles, metric = train_parameters(hourly_by_device)
    for sensor, bundle in bundles.items():
        joblib.dump(bundle, output / f"model_parameter_{sensor}.joblib")
    for audit in audits:
        device = audit["device_id"]
        try:
            future = forecast_parameters(hourly_by_device[device], bundles, as_of=as_of)
            future = future.join(classify_rules(future, config))
            future["device_id"] = device
            forecasts.append(future.reset_index())
            audit["result"] = "ok"
        except ValueError as error:
            audit["result"] = str(error)
    forecast = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    audit_table = pd.DataFrame(audits)
    forecast.to_csv(output / "prediksi_24_jam.csv", index=False)
    metric.to_csv(output / "evaluasi_mae.csv", index=False)
    audit_table.to_csv(output / "audit_perangkat.csv", index=False)
    return forecast, metric, audit_table


if __name__ == "__main__":
    config = json.loads(Path("aturan_risiko.json").read_text(encoding="utf-8"))
    raw, coverage = load_mysql()
    print("Cakupan waktu:", coverage.to_dict("records"))
    forecast, metric, audit = run_all(raw, config)
    print(audit[["device_id", "result"]].to_string(index=False))
    print(f"Tersimpan {len(forecast)} baris prediksi di hasil_ews/. Klasifikasi: {config['enabled']}.")
