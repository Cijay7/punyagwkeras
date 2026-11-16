"""Jalankan di komputer/backend MySQL, bukan di dalam Hugging Face Space."""
import json
import os
import secrets
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

bearer = HTTPBearer(auto_error=False)


def authenticate(credentials: HTTPAuthorizationCredentials | None = Security(bearer)):
    expected = os.getenv("SOURCE_API_TOKEN", "")
    if not expected or credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(401, "Bearer token sumber tidak valid.")


@asynccontextmanager
async def lifespan(app):
    if not os.getenv("SOURCE_API_TOKEN"):
        raise RuntimeError("Set SOURCE_API_TOKEN sebelum menjalankan bridge sumber.")
    url = make_url(os.getenv("MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir")).set(
        database=os.getenv("SOURCE_DATABASE", "tugasakhir_sintetis_10menit"))
    app.state.engine = create_engine(url, pool_pre_ping=True, hide_parameters=True,
        connect_args={"connect_timeout": 5, "read_timeout": 15, "init_command": "SET time_zone = '+00:00'"})
    try:
        yield
    finally:
        app.state.engine.dispose()


app = FastAPI(title="EWS Read-only Source Bridge", lifespan=lifespan)


@app.exception_handler(SQLAlchemyError)
async def db_error(request, error):
    return JSONResponse(status_code=503, content={"error": "Database sumber tidak tersedia."})


def metadata(conn):
    value = conn.execute(text("SELECT details FROM dataset_metadata WHERE id=1")).scalar_one()
    result = json.loads(value) if isinstance(value, str) else value
    return result


@app.get("/health")
def health():
    with app.state.engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/v1/devices", dependencies=[Depends(authenticate)])
def devices():
    with app.state.engine.connect() as conn:
        zone = metadata(conn)["timezone"]
        now = pd.Timestamp.now(tz=zone).tz_localize(None)
        frame = pd.read_sql(text("""SELECT device_id,MAX(date_time) AS latest_time FROM sensor_data
            WHERE device_id IS NOT NULL AND date_time<=:now GROUP BY device_id ORDER BY device_id"""), conn, params={"now": now.to_pydatetime()})
    return {"devices": [{"device_id": row.device_id, "latest_time": pd.Timestamp(row.latest_time).tz_localize(zone).isoformat()}
                        for row in frame.itertuples(index=False)]}


@app.get("/api/v1/devices/{device_id}/history", dependencies=[Depends(authenticate)])
def history(device_id: str = Path(pattern=r"^[A-Za-z0-9_-]{1,100}$"), hours: int = Query(25, ge=24, le=48)):
    with app.state.engine.connect() as conn:
        zone = metadata(conn)["timezone"]
        now = pd.Timestamp.now(tz=zone).tz_localize(None)
        latest = conn.execute(text("""SELECT date_time,id,time_is_synthetic FROM sensor_data
            WHERE device_id=:device AND date_time<=:now ORDER BY date_time DESC,id DESC LIMIT 1"""),
            {"device": device_id, "now": now.to_pydatetime()}).first()
        if latest is None:
            raise HTTPException(404, "Device belum mempunyai pembacaan.")
        start = pd.Timestamp(latest.date_time).floor("h")-pd.Timedelta(hours=hours)
        frame = pd.read_sql(text("""SELECT date_time,soil_moisture AS SM,soil_temperature AS ST,
            ph_value AS ph,soil_conductivity AS SEC,water_level AS WL,rain_meter AS RR
            FROM sensor_data WHERE device_id=:device AND date_time>=:start
            AND (date_time<:latest OR (date_time=:latest AND id<=:last_id))
            ORDER BY date_time,id LIMIT 2001"""), conn, params={"device": device_id, "start": start.to_pydatetime(),
            "latest": latest.date_time, "last_id": int(latest.id)})
    if len(frame)>2000:
        raise HTTPException(503, "Histori melebihi batas 2000 baris; atur sumber sebelum inferensi.")
    readings = []
    for row in frame.itertuples(index=False):
        readings.append({"date_time": pd.Timestamp(row.date_time).tz_localize(zone).isoformat(),
                         **{sensor: float(getattr(row,sensor)) if getattr(row,sensor) is not None and np.isfinite(getattr(row,sensor)) else None
                            for sensor in ["SM","ST","ph","SEC","WL","RR"]}})
    return {"device_id": device_id, "time_is_synthetic": bool(latest.time_is_synthetic), "readings": readings}
