"""REST API EWS: baca terbaru dari MySQL, inferensi tanpa melatih ulang."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Path as ApiPath, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ews_gambut import (FEATURES, HORIZON, LAGS, classify_rules,
                        predict_parameters, prepare_device, validate_rules)

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("ews.api")
DeviceId = Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9_-]{1,100}$")]


@dataclass(frozen=True)
class Settings:
    mysql_url: str
    model_dir: Path
    rules_path: Path
    timezone: str = "Asia/Jakarta"
    max_age_minutes: int = 120
    max_history_rows: int = 50000

    @classmethod
    def from_env(cls):
        value = cls(
            mysql_url=os.getenv("MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir"),
            model_dir=Path(os.getenv("EWS_MODEL_DIR", str(ROOT / "hasil_ews"))),
            rules_path=Path(os.getenv("EWS_RULES_PATH", str(ROOT / "aturan_risiko.json"))),
            timezone=os.getenv("DB_TIMEZONE", "Asia/Jakarta"),
            max_age_minutes=int(os.getenv("EWS_MAX_AGE_MINUTES", "120")),
            max_history_rows=int(os.getenv("EWS_MAX_HISTORY_ROWS", "50000")),
        )
        ZoneInfo(value.timezone)
        if value.max_age_minutes < 1 or value.max_history_rows < LAGS:
            raise ValueError("Pengaturan umur data/batas baris tidak valid.")
        return value


class ApiError(Exception):
    def __init__(self, code, message, status_code=503):
        self.code, self.message, self.status_code = code, message, status_code


class SensorValues(BaseModel):
    SM: float | None
    ST: float | None
    ph: float | None
    SEC: float | None
    WL: float | None
    RR: float | None


class Classification(BaseModel):
    status: str
    alasan: str
    basis: str = "aturan_ambang"
    aturan_aktif: bool


class LatestReading(BaseModel):
    id: int
    date_time: str
    age_minutes: float
    data_lama: bool
    suspect_zero_packet: bool
    sensors: SensorValues
    classification: Classification


class LatestResponse(BaseModel):
    device_id: str
    checked_at: str
    timezone: str
    latest: LatestReading


class ForecastPoint(BaseModel):
    date_time: str
    horizon_jam: int
    sensors: SensorValues
    classification: Classification


class Prediction(BaseModel):
    availability: Literal["ready", "stale_data", "invalid_latest_reading", "insufficient_history",
                          "model_not_found", "model_invalid", "model_from_future"]
    message: str
    horizon_hours: int = HORIZON
    forecast_origin: str | None = None
    model_trained_until: str | None = None
    method_per_sensor: dict[str, str] | None = None
    method_per_horizon: dict[str, list[str]] | None = None
    evaluation_warning: str | None = None
    points: list[ForecastPoint] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    device_id: str
    checked_at: str
    timezone: str
    prediction: Prediction


class ResultResponse(LatestResponse):
    prediction: Prediction


class MySQLRepository:
    """Koneksi pool; seluruh kueri baca saja dan dibatasi perangkat/waktu."""

    def __init__(self, settings):
        self.max_history_rows = settings.max_history_rows
        self.engine = create_engine(
            settings.mysql_url, pool_pre_ping=True, pool_recycle=1800,
            hide_parameters=True,
            connect_args={"connect_timeout": 5, "read_timeout": 15, "write_timeout": 15},
        )

    def close(self):
        self.engine.dispose()

    def health(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1")).scalar_one()

    def devices(self, now):
        with self.engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT device_id, MAX(date_time) AS latest_time
                FROM sensor_data
                WHERE device_id IS NOT NULL AND date_time <= :now
                GROUP BY device_id ORDER BY device_id
            """), conn, params={"now": now.to_pydatetime()})

    def latest(self, device_id, now):
        with self.engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT id, device_id, date_time,
                       soil_moisture AS SM, soil_temperature AS ST, ph_value AS ph,
                       soil_conductivity AS SEC, water_level AS WL, rain_meter AS RR
                FROM sensor_data
                WHERE device_id = :device AND date_time <= :now
                ORDER BY date_time DESC, id DESC LIMIT 1
            """), conn, params={"device": device_id, "now": now.to_pydatetime()})

    def history(self, device_id, latest_row):
        latest_time = pd.Timestamp(latest_row["date_time"])
        # 24 bin jam lengkap + jam parsial terbaru. Tidak membaca histori pelatihan.
        start = latest_time.floor("h") - pd.Timedelta(hours=LAGS)
        with self.engine.connect() as conn:
            frame = pd.read_sql(text("""
                SELECT id, device_id, date_time,
                       soil_moisture AS SM, soil_temperature AS ST, ph_value AS ph,
                       soil_conductivity AS SEC, water_level AS WL, rain_meter AS RR
                FROM sensor_data
                WHERE device_id = :device AND date_time >= :start
                  AND (date_time < :latest_time OR (date_time = :latest_time AND id <= :latest_id))
                ORDER BY date_time, id LIMIT :row_limit
            """), conn, params={"device": device_id, "start": start.to_pydatetime(),
                               "latest_time": latest_time.to_pydatetime(),
                               "latest_id": int(latest_row["id"]),
                               "row_limit": self.max_history_rows + 1})
        if len(frame) > self.max_history_rows:
            raise ApiError("history_limit_exceeded", "Data melebihi EWS_MAX_HISTORY_ROWS; naikkan batas sebelum inferensi.")
        return frame


class EwsService:
    def __init__(self, repository, settings, clock=None):
        self.repository, self.settings = repository, settings
        self.clock = clock or (lambda: pd.Timestamp.now(tz=settings.timezone).tz_localize(None))
        self.lock = RLock()
        # Cache hanya komputasi/model. Data sensor dan aturan selalu dibaca ulang.
        self.models = OrderedDict()
        self.predictions = OrderedDict()

    def iso(self, value):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(self.settings.timezone)
        return stamp.isoformat()

    def rules(self):
        try:
            rules = json.loads(self.settings.rules_path.read_text(encoding="utf-8"))
            if not isinstance(rules, dict) or not isinstance(rules.get("enabled"), bool):
                raise ValueError("Konfigurasi harus berupa object dengan enabled boolean.")
            if rules["enabled"] and not isinstance(rules.get("rules"), list):
                raise ValueError("rules harus berupa daftar.")
            validate_rules(rules)
            return rules
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            logger.warning("Aturan risiko tidak dapat dipakai: %s", type(error).__name__)
            raise ApiError("invalid_rules", "Perbaiki konfigurasi aturan_risiko.json; ambang aktif harus lengkap.") from None

    def classify(self, values, rules, *, stale=False, suspect=False):
        if stale:
            status, reason = "Data lama", "Perbarui data sensor sebelum menentukan peringatan saat ini"
        elif suspect:
            status, reason = "Data tidak valid", "Paket SM/ST/ph/SEC nol bersamaan; periksa alat"
        else:
            result = classify_rules(pd.DataFrame([values]), rules).iloc[0]
            status, reason = result.status, result.alasan
        return {"status": status, "alasan": reason, "basis": "aturan_ambang",
                "aturan_aktif": bool(rules["enabled"])}

    def latest(self, device_id, now=None):
        now = self.clock() if now is None else now
        rules = self.rules()
        frame = self.repository.latest(device_id, now)
        if frame.empty:
            raise ApiError("device_not_found", "Perangkat belum memiliki pembacaan dengan waktu valid sampai saat ini.", 404)
        row = frame.iloc[0]
        numeric = pd.to_numeric(row[FEATURES], errors="coerce")
        values = {key: float(numeric[key]) if np.isfinite(numeric[key]) else None for key in FEATURES}
        age = max(0.0, (now - pd.Timestamp(row.date_time)).total_seconds() / 60)
        stale = age > self.settings.max_age_minutes
        suspect = all(values[key] == 0 for key in ("SM", "ST", "ph", "SEC"))
        response = {
            "device_id": device_id, "checked_at": self.iso(now), "timezone": self.settings.timezone,
            "latest": {"id": int(row.id), "date_time": self.iso(row.date_time),
                       "age_minutes": round(age, 2), "data_lama": stale,
                       "suspect_zero_packet": suspect, "sensors": values,
                       "classification": self.classify(values, rules, stale=stale, suspect=suspect)},
        }
        return response, row, rules

    def _model(self):
        paths = [self.settings.model_dir / f"model_parameter_{sensor}.joblib" for sensor in FEATURES]
        try:
            key = tuple((str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        except FileNotFoundError:
            return None, None
        with self.lock:
            if key in self.models:
                self.models.move_to_end(key)
                return self.models[key], key
            bundles = {}
            for sensor, path in zip(FEATURES, paths):
                # File model hanya berasal dari direktori lokal yang dipercaya.
                bundle = joblib.load(path)
                if not isinstance(bundle, dict) or not isinstance(bundle.get("metadata"), dict):
                    raise ValueError("Format bundle tidak valid.")
                meta = bundle["metadata"]
                if (meta.get("training_scope") != "parameter" or meta.get("sensor") != sensor or
                        meta.get("lag_hours") != LAGS or meta.get("horizon_hours") != HORIZON):
                    raise ValueError("Parameter atau dimensi model tidak sesuai.")
                if meta.get("method") not in ("Ridge", "Nilai terakhir", "Seleksi per horizon"):
                    raise ValueError("Metode sensor tidak dikenal.")
                if not hasattr(bundle.get("model"), "predict") or pd.isna(pd.Timestamp(meta.get("trained_until"))):
                    raise ValueError("Model atau waktu pelatihan tidak tersedia.")
                bundles[sensor] = bundle
            combined = {"parameters": bundles, "metadata": {
                "forecast_origin": str(max(pd.Timestamp(b["metadata"]["trained_until"]) for b in bundles.values())),
                "method_per_sensor": {sensor: b["metadata"]["method"] for sensor, b in bundles.items()},
                "method_per_horizon": {sensor: b["metadata"].get("method_per_horizon", [b["metadata"]["method"]] * HORIZON)
                                       for sensor, b in bundles.items()},
                "evaluation_warning": "; ".join(dict.fromkeys(
                    b["metadata"].get("evaluation_warning", "Model eksperimen; lihat evaluasi_mae.csv.")
                    for b in bundles.values()))}}
            self.models[key] = combined
            while len(self.models) > 32:
                self.models.popitem(last=False)
            return combined, key

    def results(self, device_id):
        now = self.clock()
        response, row, rules = self.latest(device_id, now)
        prediction = {"availability": "stale_data", "message": "Data terbaru sudah lama; menunggu telemetri baru.",
                      "horizon_hours": HORIZON, "points": []}
        response["prediction"] = prediction
        if response["latest"]["data_lama"]:
            return response
        if response["latest"]["suspect_zero_packet"]:
            prediction.update(availability="invalid_latest_reading", message="Paket sensor terbaru mencurigakan; periksa alat.")
            return response
        try:
            bundle, model_key = self._model()
        except Exception as error:
            logger.warning("Gagal membaca model %s: %s", device_id, type(error).__name__)
            prediction.update(availability="model_invalid", message="File model tidak valid; latih ulang secara offline.")
            return response
        if bundle is None:
            prediction.update(availability="model_not_found", message="Model per parameter belum lengkap; latih melalui notebook atau python ews_gambut.py.")
            return response
        raw = self.repository.history(device_id, row)
        hourly, _ = prepare_device(raw, device_id)
        origin = pd.Timestamp(row.date_time).floor("h")
        required = pd.date_range(origin - pd.Timedelta(hours=LAGS - 1), periods=LAGS, freq="h")
        window = hourly.reindex(required)
        if window.isna().any(axis=None):
            prediction.update(availability="insufficient_history", message="Perlu 24 jam pengukuran lengkap terakhir; terdapat jam/sensor yang kosong.")
            return response
        prediction["forecast_origin"] = self.iso(origin)
        if (now - origin).total_seconds() / 60 > self.settings.max_age_minutes:
            prediction.update(message="Jam input lengkap terakhir sudah lama; menunggu pengukuran baru.")
            return response
        meta = bundle["metadata"]
        trained_until = pd.Timestamp(meta["forecast_origin"])
        prediction["model_trained_until"] = self.iso(trained_until)
        if trained_until > origin:
            prediction.update(availability="model_from_future", message="Model dilatih melewati waktu input; gunakan model dengan waktu pelatihan yang sesuai.")
            return response
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(window, index=True).values.tobytes()).hexdigest()
        key = (device_id, model_key, fingerprint)
        try:
            with self.lock:
                future = self.predictions.get(key)
                if future is None:
                    future = predict_parameters(window, bundle["parameters"])
                    if future.shape != (HORIZON, len(FEATURES)) or not np.isfinite(future).all():
                        raise ValueError("Keluaran model tidak valid.")
                    self.predictions[key] = future
                    while len(self.predictions) > 128:
                        self.predictions.popitem(last=False)
                else:
                    self.predictions.move_to_end(key)
        except Exception as error:
            logger.warning("Inferensi gagal %s: %s", device_id, type(error).__name__)
            prediction.update(availability="model_invalid", message="Inferensi model gagal; periksa kecocokan model dan versi dependensi.")
            return response
        points = []
        for h, values in enumerate(future, start=1):
            sensors = dict(zip(FEATURES, map(float, values)))
            points.append({"date_time": self.iso(origin + pd.Timedelta(hours=h)), "horizon_jam": h,
                           "sensors": sensors, "classification": self.classify(sensors, rules)})
        prediction.update(availability="ready", message="Prediksi dari 24 jam input lengkap terbaru; bukan probabilitas kebakaran.",
                          method_per_sensor=meta["method_per_sensor"],
                          method_per_horizon=meta["method_per_horizon"],
                          evaluation_warning=meta.get("evaluation_warning", "Model eksperimen; lihat evaluasi_mae.csv. Akurasi risiko kebakaran belum diuji."),
                          points=points)
        return response


def create_app(settings=None, repository=None, clock=None):
    settings = settings or Settings.from_env()
    owned_repository = repository is None

    @asynccontextmanager
    async def lifespan(application):
        repo = MySQLRepository(settings) if owned_repository else repository
        application.state.service = EwsService(repo, settings, clock)
        try:
            yield
        finally:
            if owned_repository:
                repo.close()

    application = FastAPI(
        title="EWS Lahan Gambut API", version="1.0.0", lifespan=lifespan,
        description="Baca MySQL terbaru pada setiap request. Prediksi memakai 24 jam input terakhir dan model offline.",
    )

    @application.middleware("http")
    async def no_http_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(ApiError)
    async def api_error_handler(request, error):
        return JSONResponse(status_code=error.status_code,
                            content={"error": {"code": error.code, "message": error.message}})

    @application.exception_handler(SQLAlchemyError)
    async def database_error_handler(request, error):
        logger.warning("Koneksi/kueri database gagal: %s", type(error).__name__)
        return JSONResponse(status_code=503, content={"error": {
            "code": "database_unavailable", "message": "MySQL tidak tersedia; periksa layanan dan MYSQL_URL."}})

    @application.get("/health", tags=["Status"])
    def health(request: Request):
        request.app.state.service.repository.health()
        return {"status": "ok", "database": "connected"}

    @application.get("/api/v1/devices", tags=["Perangkat"])
    def devices(request: Request):
        service = request.app.state.service
        now = service.clock()
        frame = service.repository.devices(now)
        return {"checked_at": service.iso(now), "timezone": settings.timezone, "devices": [
            {"device_id": row.device_id, "latest_time": service.iso(row.latest_time),
             "data_lama": (now - pd.Timestamp(row.latest_time)).total_seconds() / 60 > settings.max_age_minutes}
            for row in frame.itertuples(index=False)]}

    @application.get("/api/v1/devices/{device_id}/latest", response_model=LatestResponse, tags=["Sensor"])
    def latest(device_id: DeviceId, request: Request):
        return request.app.state.service.latest(device_id)[0]

    @application.get("/api/v1/devices/{device_id}/classification", tags=["Klasifikasi"])
    def classification(device_id: DeviceId, request: Request):
        response = request.app.state.service.latest(device_id)[0]
        return {"device_id": device_id, "checked_at": response["checked_at"],
                "date_time": response["latest"]["date_time"],
                "data_lama": response["latest"]["data_lama"],
                **response["latest"]["classification"]}

    @application.get("/api/v1/devices/{device_id}/predictions", response_model=PredictionResponse, tags=["Prediksi"])
    def predictions(device_id: DeviceId, request: Request):
        response = request.app.state.service.results(device_id)
        return {key: response[key] for key in ("device_id", "checked_at", "timezone", "prediction")}

    @application.get("/api/v1/devices/{device_id}/results", response_model=ResultResponse, tags=["Prediksi"])
    def results(device_id: DeviceId, request: Request):
        return request.app.state.service.results(device_id)

    return application


app = create_app()
