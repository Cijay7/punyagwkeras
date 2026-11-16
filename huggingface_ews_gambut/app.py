"""FastAPI mandiri untuk Hugging Face Docker Space, port 7860."""
from __future__ import annotations

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path as ApiPath, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader

from core import FEATURES, History, Mode, Unavailable, classify, prepare, validate_rules
from source import Source, SourceError

ROOT = Path(__file__).resolve().parent
DeviceId = Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9_-]{1,100}$")]
logger = logging.getLogger("ews.space")


def create_app(models=None, source=None, clock=None, api_key=None):
    timezone = os.getenv("TIMEZONE", "Asia/Jakarta")
    ZoneInfo(timezone)
    key = os.getenv("API_KEY", "") if api_key is None else api_key
    header = APIKeyHeader(name="X-API-Key", auto_error=False)
    age_limit = int(os.getenv("MAX_AGE_MINUTES", "120"))
    if age_limit < 1:
        raise ValueError("MAX_AGE_MINUTES harus positif.")

    def authenticate(value=Security(header)):
        if key and (value is None or not secrets.compare_digest(value, key)):
            raise HTTPException(401, "X-API-Key tidak valid.")

    @asynccontextmanager
    async def lifespan(application):
        if models is None:
            from runtime import Models
            application.state.models = Models(ROOT/"models", timezone)
        else:
            application.state.models = models
        application.state.source = source or Source(os.getenv("SOURCE_API_BASE_URL", ""), os.getenv("SOURCE_API_TOKEN", ""),
            os.getenv("DATA_SOURCE", "https") == "example", ROOT/"examples"/"readings.json")
        rules = json.loads((ROOT/"aturan_risiko.json").read_text(encoding="utf-8"))
        validate_rules(rules)
        application.state.rules = rules
        yield

    app = FastAPI(title="EWS Gambut - Keras REST", version="1.0.0", lifespan=lifespan,
                  description="GET mengambil histori dari REST API sumber HTTPS; POST menerima histori langsung. Model dilatih pada waktu sintetis. Mode demo bukan peringatan live atau backtest.")
    origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    @app.middleware("http")
    async def no_cache(request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > 2_000_000:
            return JSONResponse(status_code=413, content={"error": {"code": "body_too_large", "message": "Batas body 2 MB."}})
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(SourceError)
    async def source_error(request, error):
        return JSONResponse(status_code=error.status, content={"error": {"code": error.code, "message": error.message}})

    def now():
        return clock() if clock else pd.Timestamp.now(tz=timezone).tz_localize(None)

    def iso(value):
        return pd.Timestamp(value).tz_localize(timezone).isoformat()

    def latest_view(history, current, rules):
        reading = max(history.readings, key=lambda row: row.date_time)
        stamp = pd.Timestamp(reading.date_time).tz_convert(timezone).tz_localize(None)
        values = {s: getattr(reading, s) for s in FEATURES}
        age = max(0, (current-stamp).total_seconds()/60)
        suspect = all(values[s] == 0 for s in ("SM", "ST", "ph", "SEC"))
        if age > age_limit:
            classification = classify(values, rules, True)
        elif suspect or any(value is None for value in values.values()):
            classification = {"status": "Data tidak lengkap/valid", "alasan": "Periksa paket sensor terbaru"}
        else:
            classification = classify(values, rules, False)
        return {"date_time": iso(stamp), "age_minutes": round(age, 2), "data_lama": age > age_limit,
                "time_is_synthetic": history.time_is_synthetic, "suspect_zero_packet": suspect,
                "sensors": values, "classification": classification}

    def compute(device_id, history, mode, request, data_source):
        if history.device_id != device_id:
            raise HTTPException(422, "device_id body harus sama dengan path.")
        current = now()
        prediction = {"availability": "insufficient_history", "message": "Input belum cukup.", "mode": mode,
                      "is_demo": mode == "demo", "is_backtest": False, "is_operational": False,
                      "time_is_synthetic": history.time_is_synthetic, "model_trained_on_synthetic": True,
                      "horizon_hours": 24, "points": [],
                      "evaluation_warning": "Model waktu sintetis; bukan bukti akurasi lapangan atau kebakaran."}
        response = {"device_id": device_id, "checked_at": iso(current), "timezone": timezone,
                    "data_source": data_source, "time_is_synthetic": history.time_is_synthetic,
                    "latest": latest_view(history, current, request.app.state.rules), "prediction": prediction}
        try:
            window, origin = prepare(history, timezone, current)
        except Unavailable as error:
            prediction.update(availability=error.code, message=error.message)
            return response
        stale = current-origin > pd.Timedelta(minutes=age_limit)
        info = request.app.state.models.info(origin)
        after = any(item["training_after_input"] for item in info.values())
        prediction.update(forecast_origin=iso(origin), input_data_lama=bool(stale),
                          input_age_minutes=round(max(0, (current-origin).total_seconds()/60), 2),
                          model_per_parameter=info, model_training_after_input=after)
        if mode == "live":
            if stale:
                prediction.update(availability="stale_data", message="Mode live memerlukan input segar.")
            else:
                prediction.update(availability="synthetic_model", message="Champion dilatih pada waktu sintetis; belum untuk peringatan live.")
            return response
        try:
            future = request.app.state.models.predict(window)
        except Exception as error:
            logger.warning("Inferensi gagal: %s", type(error).__name__)
            raise HTTPException(503, "Inferensi Keras gagal; periksa checkpoint dan dependensi.") from None
        for h, values in enumerate(future, start=1):
            sensors = dict(zip(FEATURES, map(float, values)))
            prediction["points"].append({"date_time": iso(origin+pd.Timedelta(hours=h)), "horizon_jam": h,
                                          "sensors": sensors, "classification": classify(sensors, request.app.state.rules, stale)})
        prediction.update(availability="demo_ready", message="24 titik demo dari champion Keras; bukan kondisi saat ini atau backtest.")
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "models_loaded": len(request.app.state.models.info()),
                "source_configured": request.app.state.source.configured, "model_trained_on_synthetic": True}

    @app.get("/api/v1/models", dependencies=[Depends(authenticate)])
    def model_info(request: Request):
        return {"framework": "keras", "models": request.app.state.models.info(), "model_trained_on_synthetic": True}

    @app.get("/api/v1/devices", dependencies=[Depends(authenticate)])
    def devices(request: Request):
        return request.app.state.source.devices()

    @app.get("/api/v1/devices/{device_id}/predictions", dependencies=[Depends(authenticate)])
    def get_predictions(device_id: DeviceId, request: Request, mode: Mode = Query("demo")):
        provider = request.app.state.source
        history = provider.history(device_id)
        return compute(device_id, history, mode, request, "bundled_example" if provider.example else "https_rest_api")

    @app.get("/api/v1/devices/{device_id}/results", dependencies=[Depends(authenticate)])
    def results(device_id: DeviceId, request: Request, mode: Mode = Query("demo")):
        return get_predictions(device_id, request, mode)

    @app.get("/api/v1/devices/{device_id}/latest", dependencies=[Depends(authenticate)])
    def latest(device_id: DeviceId, request: Request):
        history = request.app.state.source.history(device_id)
        current = now()
        return {"device_id": device_id, "checked_at": iso(current), "timezone": timezone,
                "latest": latest_view(history, current, request.app.state.rules)}

    @app.get("/api/v1/devices/{device_id}/classification", dependencies=[Depends(authenticate)])
    def classification(device_id: DeviceId, request: Request):
        result = latest(device_id, request)
        return {"device_id": device_id, "date_time": result["latest"]["date_time"],
                "time_is_synthetic": result["latest"]["time_is_synthetic"], **result["latest"]["classification"]}

    @app.post("/api/v1/devices/{device_id}/predictions", dependencies=[Depends(authenticate)])
    def post_predictions(device_id: DeviceId, history: History, request: Request, mode: Mode = Query("demo")):
        return compute(device_id, history, mode, request, "request_json")

    return app


app = create_app()
