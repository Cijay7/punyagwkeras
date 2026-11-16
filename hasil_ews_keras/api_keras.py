"""API lokal Keras: enam champion parameter, input device pilihan, demo sintetis."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from api import (ApiError, DeviceId, EwsService, ForecastPoint, LatestReading,
                 LatestResponse, MySQLRepository, Settings)
from ews_gambut import FEATURES, HORIZON, LAGS, predictors, prepare_device

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("ews.keras_api")
Mode = Literal["demo", "live"]
Availability = Literal["demo_ready", "ready", "stale_data", "synthetic_data", "invalid_latest_reading",
                       "insufficient_history", "model_not_found", "model_invalid", "model_from_future"]


@dataclass(frozen=True)
class KerasSettings(Settings):
    @classmethod
    def from_env(cls):
        configured = os.getenv("EWS_KERAS_MYSQL_URL")
        url = configured or make_url(os.getenv("MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir")).set(
            database=os.getenv("EWS_KERAS_DATABASE", "tugasakhir_sintetis_10menit"))
        value = cls(mysql_url=str(url) if isinstance(url, str) else url.render_as_string(hide_password=False),
                    model_dir=Path(os.getenv("EWS_KERAS_MODEL_DIR", str(ROOT / "hasil_ews_keras"))),
                    rules_path=Path(os.getenv("EWS_KERAS_RULES_PATH", str(ROOT / "aturan_risiko.json"))),
                    timezone=os.getenv("EWS_KERAS_TIMEZONE", "Asia/Jakarta"),
                    max_age_minutes=int(os.getenv("EWS_KERAS_MAX_AGE_MINUTES", "120")),
                    max_history_rows=int(os.getenv("EWS_KERAS_MAX_HISTORY_ROWS", "50000")))
        ZoneInfo(value.timezone)
        if value.max_age_minutes < 1 or value.max_history_rows < LAGS:
            raise ValueError("Batas umur data atau baris histori tidak valid.")
        return value


class KerasLatestReading(LatestReading):
    time_is_synthetic: bool


class KerasLatestResponse(LatestResponse):
    latest: KerasLatestReading
    database: str
    time_is_synthetic: bool


class ModelInfo(BaseModel):
    source_device_id: str
    framework: str = "keras"
    architecture: str = "GRU"
    trained_until: str
    training_after_input: bool = False
    validation_macro_mae: float | None = None
    test_macro_mae: float | None = None
    test_baseline_macro_mae: float | None = None


class KerasPrediction(BaseModel):
    availability: Availability
    message: str
    mode: Mode
    is_demo: bool
    is_backtest: bool = False
    is_operational: bool = False
    time_is_synthetic: bool
    horizon_hours: int = HORIZON
    forecast_origin: str | None = None
    input_data_lama: bool
    input_age_minutes: float | None = None
    model_trained_until: str | None = None
    model_training_after_input: bool = False
    method_per_sensor: dict[str, str] | None = None
    model_per_parameter: dict[str, ModelInfo] | None = None
    evaluation_warning: str
    points: list[ForecastPoint] = Field(default_factory=list)


class KerasPredictionResponse(BaseModel):
    device_id: str
    checked_at: str
    timezone: str
    database: str
    time_is_synthetic: bool
    prediction: KerasPrediction


class KerasResultResponse(KerasLatestResponse):
    prediction: KerasPrediction


class KerasRepository(MySQLRepository):
    def metadata(self):
        with self.engine.connect() as conn:
            value = conn.execute(text("SELECT details FROM dataset_metadata WHERE id=1")).scalar_one()
        return json.loads(value) if isinstance(value, str) else value

    def latest(self, device_id, now):
        with self.engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT id,device_id,date_time,time_is_synthetic,
                       soil_moisture AS SM,soil_temperature AS ST,ph_value AS ph,
                       soil_conductivity AS SEC,water_level AS WL,rain_meter AS RR
                FROM sensor_data WHERE device_id=:device AND date_time<=:now
                ORDER BY date_time DESC,id DESC LIMIT 1
            """), conn, params={"device": device_id, "now": now.to_pydatetime()})


def load_keras(path):
    # TensorFlow diimpor hanya saat model pertama diminta.
    from ews_gambut_keras import load_bundle
    return load_bundle(path)


def predict_keras(model, x, sensor, scale):
    from ews_gambut_keras import predict
    return predict(model, x, sensor, scale)


class KerasService(EwsService):
    def __init__(self, repository, settings, clock=None, model_loader=None, predictor=None):
        super().__init__(repository, settings, clock)
        self.model_loader = model_loader or load_keras
        self.predictor = predictor or predict_keras

    def dataset(self):
        try:
            meta = self.repository.metadata()
            if (not isinstance(meta, dict) or meta.get("time_semantics") != "synthetic_not_actual_measurement" or
                    meta.get("timezone") != self.settings.timezone):
                raise ValueError("Dataset atau zona waktu berbeda.")
            return meta
        except (ValueError, TypeError, KeyError) as error:
            raise ApiError("dataset_mismatch", "API ini memerlukan dataset sintetis dan timezone sesuai metadata.") from None

    def latest(self, device_id, now=None):
        self.dataset()
        response, row, rules = super().latest(device_id, now)
        synthetic = bool(row.time_is_synthetic)
        if not synthetic:
            raise ApiError("dataset_mismatch", "Baris input tidak ditandai sintetis sesuai metadata dataset.")
        response.update(database=make_url(self.settings.mysql_url).database, time_is_synthetic=synthetic)
        response["latest"]["time_is_synthetic"] = synthetic
        return response, row, rules

    def _model(self):
        root = self.settings.model_dir.resolve()
        manifest_path = root / "model_terbaik.json"
        if not manifest_path.exists():
            return None, None
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        paths = []
        for sensor in FEATURES:
            record = manifest.get(sensor, {})
            if record.get("status") != "selected":
                return None, None
            path = (root / record["champion_path"]).resolve()
            if path != root / "champions" / f"model_{sensor}.keras":
                raise ValueError("Lokasi champion tidak valid.")
            paths.extend([path, path.with_suffix(".json")])
        try:
            key = (hashlib.sha256(manifest_bytes).hexdigest(),
                   tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths))
        except FileNotFoundError:
            return None, None
        with self.lock:
            if key in self.models:
                return self.models[key], key
            bundles = {}
            for j, sensor in enumerate(FEATURES):
                model, bundle = self.model_loader(paths[j * 2])
                meta = bundle["metadata"]
                source_device = manifest[sensor]["device_id"]
                stamp = pd.Timestamp(meta["trained_until"])
                if (meta.get("framework") != "keras" or meta.get("architecture") != "GRU" or
                        meta.get("sensor") != sensor or meta.get("training_scope") != "device_parameter" or
                        meta.get("device_id") != source_device or meta.get("training_devices") != [source_device] or
                        meta.get("lag_hours") != LAGS or meta.get("horizon_hours") != HORIZON or
                        meta.get("time_semantics") != "synthetic_not_actual_measurement" or
                        pd.isna(stamp) or stamp.tzinfo is not None or
                        not np.isfinite(bundle["scale"]) or bundle["scale"] <= 0):
                    raise ValueError("Model tidak sesuai champion atau dimensi training.")
                for metric in ("validation_macro_mae", "test_macro_mae", "test_baseline_macro_mae"):
                    value = manifest[sensor].get(metric)
                    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or
                                              not np.isfinite(value) or value < 0):
                        raise ValueError("Metrik champion tidak valid.")
                bundles[sensor] = {"model": model, "scale": bundle["scale"], "metadata": meta,
                                   "selection": manifest[sensor]}
            # Hindari menumpuk banyak versi model TensorFlow dalam memori.
            self.models.clear()
            self.predictions.clear()
            self.models[key] = bundles
            return bundles, key

    def model_info(self, bundles, origin=None):
        return {sensor: {"source_device_id": item["metadata"]["device_id"],
                         "framework": "keras", "architecture": "GRU",
                         "trained_until": self.iso(item["metadata"]["trained_until"]),
                         "training_after_input": bool(origin is not None and pd.Timestamp(item["metadata"]["trained_until"]) > origin),
                         **{key: item["selection"].get(key) for key in
                            ("validation_macro_mae", "test_macro_mae", "test_baseline_macro_mae")}}
                for sensor, item in bundles.items()}

    def results(self, device_id, mode="demo"):
        now = self.clock()
        response, row, rules = self.latest(device_id, now)
        latest = response["latest"]
        synthetic = response["time_is_synthetic"]
        prediction = {"availability": "stale_data", "message": "Data lama; menunggu data terbaru.",
                      "mode": mode, "is_demo": mode == "demo", "is_backtest": False, "is_operational": False,
                      "time_is_synthetic": synthetic, "input_data_lama": latest["data_lama"],
                      "horizon_hours": HORIZON, "points": [],
                      "evaluation_warning": "Eksperimen waktu sintetis; pemenang validation tidak selalu mengalahkan baseline. Bukan akurasi kebakaran."}
        response["prediction"] = prediction
        if mode == "live" and latest["data_lama"]:
            return response
        if mode == "live" and synthetic:
            prediction.update(availability="synthetic_data", message="Data sintetis hanya untuk demo, bukan peringatan live.")
            return response
        if latest["suspect_zero_packet"]:
            prediction.update(availability="invalid_latest_reading", message="Paket SM/ST/ph/SEC nol bersamaan; tidak dapat membuat prediksi.")
            return response
        try:
            bundles, model_key = self._model()
        except Exception as error:
            logger.warning("Gagal memuat champion: %s", type(error).__name__)
            prediction.update(availability="model_invalid", message="Model atau metadata champion tidak valid.")
            return response
        if bundles is None:
            prediction.update(availability="model_not_found", message="Enam model champion Keras beserta metadata belum lengkap.")
            return response
        origin = pd.Timestamp(row.date_time).floor("h")
        prediction.update(forecast_origin=self.iso(origin), input_age_minutes=round(max(0, (now-origin).total_seconds()/60), 2))
        info = self.model_info(bundles, origin)
        trained_until = max(pd.Timestamp(item["metadata"]["trained_until"]) for item in bundles.values())
        after = trained_until > origin
        prediction.update(model_per_parameter=info, model_trained_until=self.iso(trained_until),
                          model_training_after_input=bool(after), method_per_sensor={s: "Keras GRU" for s in FEATURES})
        if mode == "live" and after:
            prediction.update(availability="model_from_future", message="Model dilatih melewati waktu input; gunakan model yang sesuai.")
            return response
        origin_stale = (now-origin).total_seconds()/60 > self.settings.max_age_minutes
        prediction["input_data_lama"] = bool(latest["data_lama"] or origin_stale)
        if mode == "live" and origin_stale:
            return response
        raw = self.repository.history(device_id, row)
        hourly, _ = prepare_device(raw, device_id)
        required = pd.date_range(origin-pd.Timedelta(hours=LAGS-1), periods=LAGS, freq="h")
        window = hourly.reindex(required)
        if window.isna().any(axis=None):
            prediction.update(availability="insufficient_history", message="Perlu 24 jam input lengkap terakhir; terdapat jam atau sensor kosong.")
            return response
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(window, index=True).values.tobytes()).hexdigest()
        cache_key = (device_id, model_key, fingerprint)
        try:
            with self.lock:
                future = self.predictions.get(cache_key)
                if future is None:
                    future = np.empty((HORIZON, len(FEATURES)))
                    for j, sensor in enumerate(FEATURES):
                        item = bundles[sensor]
                        values = np.asarray(self.predictor(item["model"], predictors(window[[sensor]]).iloc[[-1]], sensor, item["scale"]))
                        if values.shape != (1, HORIZON) or not np.isfinite(values).all():
                            raise ValueError("Dimensi atau nilai keluaran model tidak valid.")
                        future[:, j] = values[0]
                    self.predictions[cache_key] = future
                    while len(self.predictions) > 128:
                        self.predictions.popitem(last=False)
                else:
                    self.predictions.move_to_end(cache_key)
        except Exception as error:
            logger.warning("Inferensi Keras gagal: %s", type(error).__name__)
            prediction.update(availability="model_invalid", message="Inferensi Keras gagal; periksa model dan dependensi.")
            return response
        for h, values in enumerate(future, start=1):
            sensors = dict(zip(FEATURES, map(float, values)))
            prediction["points"].append({"date_time": self.iso(origin+pd.Timedelta(hours=h)), "horizon_jam": h,
                                         "sensors": sensors, "classification": self.classify(sensors, rules, stale=prediction["input_data_lama"])})
        if mode == "demo":
            prediction.update(availability="demo_ready", message="Demo sintetis dari input terakhir; bukan kondisi saat ini atau backtest. Periksa penanda model_training_after_input.")
        else:
            prediction.update(availability="ready", message="Prediksi 24 jam dari input terbaru.", is_operational=True)
        return response


def create_app(settings=None, repository=None, clock=None, model_loader=None, predictor=None):
    settings = settings or KerasSettings.from_env()
    owned = repository is None

    @asynccontextmanager
    async def lifespan(application):
        repo = KerasRepository(settings) if owned else repository
        application.state.service = KerasService(repo, settings, clock, model_loader, predictor)
        try:
            yield
        finally:
            if owned:
                repo.close()

    application = FastAPI(title="EWS Gambut Keras API Lokal", version="1.0.0", lifespan=lifespan,
                          description="Default mode demo untuk database sintetis. Enam champion Keras dipakai untuk device pilihan. Tidak ada training atau penulisan database pada request.")

    @application.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(ApiError)
    async def api_error(request, error):
        return JSONResponse(status_code=error.status_code, content={"error": {"code": error.code, "message": error.message}})

    @application.exception_handler(SQLAlchemyError)
    async def sql_error(request, error):
        logger.warning("MySQL gagal: %s", type(error).__name__)
        return JSONResponse(status_code=503, content={"error": {"code": "database_unavailable", "message": "MySQL tidak tersedia; periksa layanan dan konfigurasi database Keras."}})

    @application.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @application.get("/health", tags=["Status"])
    def health(request: Request):
        service = request.app.state.service
        service.repository.health()
        service.dataset()
        return {"status": "ok", "database": make_url(settings.mysql_url).database, "framework": "keras", "default_mode": "demo", "time_is_synthetic": True}

    @application.get("/api/v1/models", tags=["Model"])
    def models(request: Request):
        service = request.app.state.service
        try:
            bundles, _ = service._model()
        except Exception:
            raise ApiError("model_invalid", "Model atau metadata champion tidak valid.") from None
        if bundles is None:
            raise ApiError("model_not_found", "Enam model champion belum lengkap.")
        return {"framework": "keras", "selection": "validation_macro_mae", "horizon_hours": HORIZON,
                "time_is_synthetic": True, "models": service.model_info(bundles)}

    @application.get("/api/v1/devices", tags=["Perangkat"])
    def devices(request: Request):
        service = request.app.state.service
        service.dataset()
        now = service.clock()
        frame = service.repository.devices(now)
        return {"checked_at": service.iso(now), "timezone": settings.timezone, "time_is_synthetic": True,
                "devices": [{"device_id": row.device_id, "latest_time": service.iso(row.latest_time),
                             "data_lama": (now-pd.Timestamp(row.latest_time)).total_seconds()/60 > settings.max_age_minutes}
                            for row in frame.itertuples(index=False)]}

    @application.get("/api/v1/devices/{device_id}/latest", response_model=KerasLatestResponse, tags=["Sensor"])
    def latest(device_id: DeviceId, request: Request):
        return request.app.state.service.latest(device_id)[0]

    @application.get("/api/v1/devices/{device_id}/classification", tags=["Klasifikasi"])
    def classification(device_id: DeviceId, request: Request):
        response = request.app.state.service.latest(device_id)[0]
        return {"device_id": device_id, "checked_at": response["checked_at"], "date_time": response["latest"]["date_time"],
                "data_lama": response["latest"]["data_lama"], "time_is_synthetic": response["time_is_synthetic"],
                **response["latest"]["classification"]}

    @application.get("/api/v1/devices/{device_id}/predictions", response_model=KerasPredictionResponse, tags=["Prediksi"])
    def predictions(device_id: DeviceId, request: Request, mode: Mode = Query("demo")):
        response = request.app.state.service.results(device_id, mode)
        return {key: response[key] for key in ("device_id", "checked_at", "timezone", "database", "time_is_synthetic", "prediction")}

    @application.get("/api/v1/devices/{device_id}/results", response_model=KerasResultResponse, tags=["Prediksi"])
    def results(device_id: DeviceId, request: Request, mode: Mode = Query("demo")):
        return request.app.state.service.results(device_id, mode)

    return application


app = create_app()
