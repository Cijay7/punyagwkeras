"""Keras GRU per device-parameter pada database sintetis; tidak mengubah kode lama."""
from __future__ import annotations

import os

# Batasi pemakaian CPU dan hindari pool tf.data pada komputer lokal.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import gc
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from ews_gambut import FEATURES, HORIZON, LAGS, parameter_bounds, predictors, prepare_device, supervised


@dataclass
class Config:
    epochs: int = 30
    patience: int = 6
    hidden_size: int = 16
    batch_size: int = 128
    learning_rate: float = .001
    seed: int = 42

    def validate(self):
        if min(self.epochs, self.patience, self.hidden_size, self.batch_size) < 1:
            raise ValueError("Epoch, patience, hidden size dan batch harus positif.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate harus positif dan finite.")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(*frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update("|".join(frame.columns).encode())
    return digest.hexdigest()


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(value)):
        raise ValueError("Identifier tidak aman untuk nama file/database.")
    return str(value)


def load_synthetic(database):
    safe_name(database)
    url = make_url(os.getenv("MYSQL_URL", "mysql+pymysql://root:@localhost:3306/tugasakhir")).set(database=database)
    engine = create_engine(url, connect_args={"connect_timeout": 5, "init_command": "SET time_zone = '+00:00'"}, hide_parameters=True)
    try:
        with engine.connect() as conn:
            raw = pd.read_sql(text("""SELECT id,device_id,device_sequence,date_time,unixtime,time_is_synthetic,
                soil_moisture AS SM,soil_temperature AS ST,ph_value AS ph,
                soil_conductivity AS SEC,water_level AS WL,rain_meter AS RR
                FROM sensor_data ORDER BY device_id,device_sequence"""), conn)
            metadata = json.loads(conn.execute(text("SELECT details FROM dataset_metadata WHERE id=1")).scalar_one())
    finally:
        engine.dispose()
    validate_synthetic(raw, metadata)
    hourly, audits = {}, []
    for device in raw.device_id.unique():
        hourly[str(device)], audit = prepare_device(raw, device)
        audits.append(audit)
    return hourly, pd.DataFrame(audits), {"database": database, "raw_rows": len(raw),
                                          "raw_fingerprint": fingerprint(raw), "dataset_metadata": metadata}


def validate_synthetic(raw, metadata):
    if raw.empty or raw.device_id.isna().any() or not raw.time_is_synthetic.eq(1).all():
        raise ValueError("Dataset harus berupa salinan sintetis yang lengkap dan ditandai.")
    if metadata.get("time_semantics") != "synthetic_not_actual_measurement":
        raise ValueError("Metadata dataset sintetis tidak sesuai.")
    zone = metadata["timezone"]
    raw["date_time"] = pd.to_datetime(raw.date_time, errors="raise")
    converted = pd.to_datetime(raw.unixtime, unit="s", utc=True).dt.tz_convert(zone).dt.tz_localize(None)
    if not raw.date_time.eq(converted).all():
        raise ValueError("date_time dan unixtime tidak sesuai zona waktu metadata.")
    interval = int(metadata["interval_seconds"])
    for _, part in raw.groupby("device_id", sort=False):
        if not np.array_equal(part.device_sequence.to_numpy(), np.arange(len(part))):
            raise ValueError("device_sequence harus berurutan dari nol per device.")
        if not part.unixtime.diff().iloc[1:].eq(interval).all():
            raise ValueError("Interval waktu sintetis tidak sesuai metadata.")


def samples(hourly, sensor):
    parts = {}
    for device, frame in hourly.items():
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Indeks hourly tidak valid.")
        if len(frame) > 1 and not frame.index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Gap hourly harus berupa NaN, bukan baris yang dibuang.")
        x, y = supervised(frame[[sensor]])
        if len(x):
            parts[str(device)] = (x, y)
    if not parts:
        raise ValueError("Tidak ada jendela lengkap.")
    return (pd.concat({d: p[0] for d, p in parts.items()}, names=["device_id", "date_time"]),
            pd.concat({d: p[1] for d, p in parts.items()}, names=["device_id", "date_time"]))


def split_time(x):
    times = x.index.get_level_values("date_time")
    origins = times.unique().sort_values()
    if len(origins) < 168:
        raise ValueError("Perlu minimal 168 origin lengkap.")
    validation, test = origins[int(.6 * len(origins))], origins[int(.8 * len(origins))]
    gap = pd.Timedelta(hours=HORIZON)
    masks = {"train": times + gap < validation,
             "validation": (times >= validation) & (times + gap < test),
             "test": times >= test, "before_test": times + gap < test}
    if min(int(masks[k].sum()) for k in ("train", "validation", "test")) < 24:
        raise ValueError("Blok waktu setelah gap horizon terlalu pendek.")
    return masks, {"validation_start": str(validation), "test_start": str(test)}


def sequence(x, sensor, scale):
    lag = x[[f"{sensor}_lag{i}" for i in range(LAGS - 1, -1, -1)]].to_numpy(dtype=np.float64)
    level = lag[:, [-1]]
    times = x.index.get_level_values("date_time") if isinstance(x.index, pd.MultiIndex) else x.index
    hours = np.asarray(times.hour)[:, None] - np.arange(LAGS - 1, -1, -1)[None, :]
    inputs = np.stack([(lag - level) / scale, np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], axis=-1).astype("float32")
    if not np.isfinite(inputs).all():
        raise ValueError("Input Keras tidak lengkap atau tidak finite.")
    return inputs, level


def build_model(config):
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(LAGS, 3)),
        keras.layers.GRU(config.hidden_size, stateful=False),
        keras.layers.Dense(HORIZON, kernel_initializer="zeros", bias_initializer="zeros"),
    ])
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=config.learning_rate, weight_decay=1e-4, clipnorm=1.0),
                  loss=keras.losses.MeanAbsoluteError(), jit_compile=False)
    return model


def predict(model, x, sensor, scale):
    inputs, level = sequence(x, sensor, scale)
    if not len(x):
        raise ValueError("Tidak ada input prediksi.")
    delta = np.concatenate([np.asarray(model.predict_on_batch(inputs[i:i + 512])) for i in range(0, len(inputs), 512)])
    result = parameter_bounds(level + delta * scale, sensor)
    if result.shape != (len(x), HORIZON) or not np.isfinite(result).all():
        raise ValueError("Keluaran Keras tidak valid.")
    return result


def fit(x, y, sensor, config, validation=None, epochs=None):
    config.validate()
    if x.index.get_level_values("device_id").nunique() != 1 or not x.index.equals(y.index):
        raise ValueError("Satu fit hanya boleh menerima satu device dengan target sejajar.")
    delta = y.to_numpy() - x[f"{sensor}_lag0"].to_numpy()[:, None]
    if not np.isfinite(delta).all():
        raise ValueError("Target training tidak finite.")
    scale = max(float(np.std(delta)), 1e-6)
    inputs, _ = sequence(x, sensor, scale)
    target = (delta / scale).astype("float32")
    model = build_model(config)
    rng = np.random.default_rng(config.seed)
    best, best_epoch, weights, waiting, history = float("inf"), 0, None, 0, []
    for epoch in range(1, (config.epochs if epochs is None else epochs) + 1):
        order = rng.permutation(len(inputs))
        model.reset_metrics()
        loss = None
        for start in range(0, len(order), config.batch_size):
            index = order[start:start + config.batch_size]
            loss = float(model.train_on_batch(inputs[index], target[index], return_dict=True)["loss"])
        if loss is None or not np.isfinite(loss):
            raise ValueError("Training loss tidak finite.")
        row = {"epoch": epoch, "train_loss_scaled": loss}
        if validation is not None:
            vx, vy = validation
            score = float(np.abs(predict(model, vx, sensor, scale) - vy.to_numpy()).mean())
            row["validation_mae"] = score
            if score < best - 1e-8:
                best, best_epoch, waiting = score, epoch, 0
                weights = [value.copy() for value in model.get_weights()]
            else:
                waiting += 1
        history.append(row)
        if validation is not None and waiting >= config.patience:
            break
    if validation is not None:
        model.set_weights(weights)
    else:
        best_epoch = int(config.epochs if epochs is None else epochs)
    return model, scale, best_epoch, history


def save_bundle(path, model, scale, meta, config, history):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.keras")
    model.save(temporary)
    temporary.replace(path)
    bundle = {"scale": float(scale), "metadata": meta, "config": asdict(config), "history": history,
              "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(path.with_suffix(".json"), bundle)


def load_bundle(path):
    path = Path(path)
    bundle = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != bundle["model_sha256"]:
        raise ValueError("Model dan metadata checkpoint tidak cocok.")
    if bundle["metadata"].get("framework") != "keras" or not np.isfinite(bundle["scale"]) or bundle["scale"] <= 0:
        raise ValueError("Checkpoint Keras tidak valid.")
    return keras.models.load_model(path, compile=False, safe_mode=True), bundle


def get_stage(path, x, y, sensor, device, config, split, resume, validation=None, epochs=None):
    signature = fingerprint(x, y, *(validation if validation is not None else ()))
    if resume and Path(path).exists() and Path(path).with_suffix(".json").exists():
        try:
            model, bundle = load_bundle(path)
            meta = bundle["metadata"]
            if (meta["fit_fingerprint"] != signature or meta["training_devices"] != [device] or
                    meta["sensor"] != sensor or bundle["config"] != asdict(config) or
                    (epochs is not None and meta["best_epoch"] != epochs) or
                    any(meta[k] != v for k, v in split.items())):
                raise ValueError("Checkpoint berbeda dari konfigurasi/data.")
            print(f"    resume {Path(path).name}", flush=True)
            return model, bundle["scale"], meta["best_epoch"], bundle["history"], meta
        except Exception as error:
            print(f"    checkpoint diulang: {type(error).__name__}", flush=True)
    model, scale, best_epoch, history = fit(x, y, sensor, config, validation, epochs)
    times = x.index.get_level_values("date_time")
    meta = {"framework": "keras", "backend": "tensorflow", "architecture": "GRU", "sensor": sensor,
            "device_id": device, "training_devices": [device], "training_scope": "device_parameter",
            "lag_hours": LAGS, "horizon_hours": HORIZON, "best_epoch": best_epoch,
            "trained_until": str((times + pd.Timedelta(hours=HORIZON)).max()),
            "fit_fingerprint": signature, "time_semantics": "synthetic_not_actual_measurement", **split}
    save_bundle(path, model, scale, meta, config, history)
    return model, scale, best_epoch, history, meta


def references(x, y, masks):
    devices = x.index.get_level_values("device_id")
    sensor = x.columns[0].rsplit("_lag", 1)[0]
    result = []
    for device in devices.unique():
        tr, val = (devices == device) & masks["train"], (devices == device) & masks["validation"]
        if tr.sum() >= 24 and val.sum() >= 24:
            if np.abs(y.loc[tr].to_numpy() - x.loc[tr, f"{sensor}_lag0"].to_numpy()[:, None]).mean() > 1e-8:
                result.append(str(device))
    return result


def score(x, y, predicted, sensor):
    devices = x.index.get_level_values("device_id")
    error = np.abs(predicted - y.to_numpy())
    baseline_error = np.abs(x[f"{sensor}_lag0"].to_numpy()[:, None] - y.to_numpy())
    rows = [{"device_id": str(d), "origins": int((devices == d).sum()),
             "mae": float(error[devices == d].mean()), "mae_baseline": float(baseline_error[devices == d].mean())}
            for d in devices.unique()]
    return {"macro_mae": float(np.mean([r["mae"] for r in rows])),
            "macro_mae_baseline": float(np.mean([r["mae_baseline"] for r in rows])), "devices": rows}


def choose(candidates):
    eligible = [c for c in candidates if c["eligible_champion"]]
    return min(eligible, key=lambda c: (c["validation_macro_mae"], c["device_id"])) if eligible else None


def train(hourly, output, config, min_train=168, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rankings, audits, manifest = [], [], {}
    for sensor in FEATURES:
        print(f"\nPARAMETER {sensor}", flush=True)
        try:
            x, y = samples(hourly, sensor)
            masks, split = split_time(x)
        except ValueError as error:
            manifest[sensor] = {"status": "unavailable", "reason": str(error)}
            audits.extend({"sensor": sensor, "device_id": d, "result": str(error)} for d in hourly)
            write_json(output / "model_terbaik.json", manifest)
            continue
        devices = x.index.get_level_values("device_id")
        refs = references(x, y, masks)
        val_common, test_common = masks["validation"] & devices.isin(refs), masks["test"] & devices.isin(refs)
        candidates = []
        for device in hourly:
            own = devices == device
            tr, val = own & masks["train"], own & masks["validation"]
            audit = {"device_id": device, "sensor": sensor, "train_windows": int(tr.sum()), "validation_windows": int(val.sum())}
            if tr.sum() < min_train or val.sum() < 24:
                audits.append(dict(audit, result="Histori train/validation tidak cukup"))
                continue
            folder = output / "devices" / safe_name(device)
            folder.mkdir(parents=True, exist_ok=True)
            constant = bool(np.abs(y.loc[tr].to_numpy() - x.loc[tr, f"{sensor}_lag0"].to_numpy()[:, None]).mean() <= 1e-8)
            cfg = Config(**asdict(config))
            if constant:
                cfg.epochs = 1
            print(f"  {device}: train={tr.sum()}, validation={val.sum()}" + (" (konstan)" if constant else ""), flush=True)
            model, scale, best, history, _ = get_stage(folder / f"validation_{sensor}.keras", x.loc[tr], y.loc[tr],
                sensor, device, cfg, split, resume, validation=(x.loc[val], y.loc[val]))
            pd.DataFrame(history).to_csv(folder / f"training_{sensor}.csv", index=False)
            result = score(x.loc[val_common], y.loc[val_common], predict(model, x.loc[val_common], sensor, scale), sensor) if val_common.any() else None
            own_mae = float(np.abs(predict(model, x.loc[val], sensor, scale) - y.loc[val].to_numpy()).mean())
            candidates.append({**audit, "folder": folder, "config": cfg, "best_epoch": best,
                               "constant_train": constant, "eligible_champion": not constant and result is not None,
                               "validation_macro_mae": result["macro_mae"] if result else None,
                               "validation_baseline_macro_mae": result["macro_mae_baseline"] if result else None,
                               "validation_own_mae": own_mae, "validation_details": result})
            del model
            gc.collect()
        # Pilih pemenang dari validation sebelum test seluruh kandidat dihitung.
        winner = choose(candidates)
        winner_device = winner["device_id"] if winner else None
        for candidate in candidates:
            device, folder, cfg, best = candidate["device_id"], candidate["folder"], candidate["config"], candidate["best_epoch"]
            own = devices == device
            before = own & masks["before_test"]
            model, scale, _, _, eval_meta = get_stage(folder / f"evaluasi_{sensor}.keras", x.loc[before], y.loc[before], sensor, device, cfg, split, resume, epochs=best)
            result = score(x.loc[test_common], y.loc[test_common], predict(model, x.loc[test_common], sensor, scale), sensor) if test_common.any() else None
            own_test = own & masks["test"]
            own_mae = float(np.abs(predict(model, x.loc[own_test], sensor, scale) - y.loc[own_test].to_numpy()).mean()) if own_test.any() else None
            del model
            gc.collect()
            final, _, _, _, _ = get_stage(folder / f"model_{sensor}.keras", x.loc[own], y.loc[own], sensor, device, cfg, split, resume, epochs=best)
            del final
            gc.collect()
            row = {k: candidate[k] for k in ("device_id", "sensor", "train_windows", "validation_windows", "best_epoch",
                    "constant_train", "eligible_champion", "validation_macro_mae", "validation_baseline_macro_mae", "validation_own_mae")}
            row.update(test_macro_mae=result["macro_mae"] if result else None,
                       test_baseline_macro_mae=result["macro_mae_baseline"] if result else None,
                       test_own_mae=own_mae, is_champion=device == winner_device,
                       model_path=str((folder / f"model_{sensor}.keras").relative_to(output)))
            rankings.append(row)
            audits.append(dict(row, result="Konstan; tidak ikut ranking" if candidate["constant_train"] else "ok"))
            write_json(folder / f"hasil_evaluasi_{sensor}.json", {"validation": candidate["validation_details"], "test": result,
                       "reference_devices": refs, "selection_uses_test": False, "evaluation_trained_until": eval_meta["trained_until"], **split})
            if device == winner_device:
                champions = output / "champions"
                champions.mkdir(exist_ok=True)
                for stage in ("model", "validation", "evaluasi"):
                    for extension in ("keras", "json"):
                        shutil.copy2(folder / f"{stage}_{sensor}.{extension}", champions / f"{stage}_{sensor}.{extension}")
                manifest[sensor] = dict(row, status="selected", champion_path=f"champions/model_{sensor}.keras",
                                       reference_devices=refs, selection_uses_test=False, **split)
                print(f"  PEMENANG {sensor}: {device}; validation={row['validation_macro_mae']}; test={row['test_macro_mae']}", flush=True)
        if winner is None:
            manifest[sensor] = {"status": "unavailable", "reason": "Tidak ada kandidat bervariasi yang layak"}
        pd.DataFrame(rankings).to_csv(output / "ranking_model.csv", index=False)
        pd.DataFrame(audits).to_csv(output / "audit_device_parameter.csv", index=False)
        write_json(output / "model_terbaik.json", manifest)
    pd.DataFrame(audits).to_csv(output / "audit_device_parameter.csv", index=False)
    write_json(output / "model_terbaik.json", manifest)
    ranking = pd.DataFrame(rankings)
    if len(ranking):
        ranking = ranking.sort_values(["sensor", "eligible_champion", "validation_macro_mae", "device_id"], ascending=[True, False, True, True])
        ranking.to_csv(output / "ranking_model.csv", index=False)
        ranking.loc[ranking.is_champion].to_csv(output / "model_terbaik.csv", index=False)
    return manifest, ranking


def forecast(hourly, output, as_of=None):
    output = Path(output)
    manifest = json.loads((output / "model_terbaik.json").read_text())
    if any(manifest.get(s, {}).get("status") != "selected" for s in FEATURES):
        raise ValueError("Belum ada enam pemenang parameter.")
    if len(hourly) < LAGS:
        raise ValueError("Perlu 24 jam input lengkap.")
    origin = hourly.index[-1]
    result = pd.DataFrame(index=pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON, freq="h"))
    result.index.name = "date_time"
    for sensor in FEATURES:
        model, bundle = load_bundle(output / manifest[sensor]["champion_path"])
        if pd.Timestamp(bundle["metadata"]["trained_until"]) > origin:
            raise ValueError("Model dilatih melewati waktu input.")
        result[sensor] = predict(model, predictors(hourly[[sensor]]).iloc[[-1]], sensor, bundle["scale"])[0]
        del model
    result["forecast_origin"] = origin
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    result["data_lama"] = bool(now - origin > pd.Timedelta(hours=2))
    result["time_is_synthetic"] = True
    return result


def report(output, manifest):
    output = Path(output)
    lines = ["# Hasil Keras pada waktu sintetis", "",
             "Model per device-parameter. Satu pemenang per parameter dipilih dari macro MAE validation pada device pembanding yang sama; test tidak memilih pemenang.", "",
             "| Parameter | Device pemenang | MAE validation | MAE test | Baseline test |", "|---|---|---:|---:|---:|"]
    fmt = lambda v: "-" if v is None else f"{v:.6f}"
    for sensor in FEATURES:
        row = manifest.get(sensor, {})
        lines.append(f"| {sensor} | {row.get('device_id', 'Belum tersedia')} | {fmt(row.get('validation_macro_mae'))} | {fmt(row.get('test_macro_mae'))} | {fmt(row.get('test_baseline_macro_mae'))} |")
    lines += ["", "Data menggunakan waktu buatan 10 menit, bukan waktu pengukuran lapangan asli. MAE ini bukan bukti akurasi kebakaran atau perbaikan pada waktu asli.",
              "Data 10 menit diagregasi menjadi rata-rata per jam, kemudian 24 jam input memprediksi 24 jam berikutnya. RR tetap rata-rata pembacaan, bukan total hujan.",
              "Pembobotan macro sama untuk tiap device pembanding. Sensor training konstan dikecualikan dari ranking. Scaler hanya memakai bagian fit pada device asal.",
              "Pemenang berarti terbaik dari kandidat layak; bukan jaminan mengalahkan baseline. Epoch dan kandidat dipilih dari validation yang sama, sehingga masih perlu konfirmasi pada data baru.",
              "File lama, model lama, dan data database tidak diubah oleh training ini."]
    (output / "hasil_training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ranking_path = output / "ranking_model.csv"
    if not ranking_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranking = pd.read_csv(ranking_path)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), layout="constrained")
    for ax, sensor in zip(axes.flat, FEATURES):
        part = ranking.loc[ranking.sensor.eq(sensor) & ranking.eligible_champion].sort_values("validation_macro_mae", ascending=False)
        if len(part):
            ax.barh(part.device_id, part.validation_macro_mae, color=["#1565c0" if win else "#777777" for win in part.is_champion])
            ax.axvline(part.validation_baseline_macro_mae.iloc[0], color="#b45309", linestyle="--", label="Baseline validation")
            ax.legend(fontsize=8)
        ax.set(title=sensor, xlabel="MAE validation macro", xlim=(0, None))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Keras GRU | waktu sintetis | biru = pemenang validation")
    fig.savefig(output / "ranking_validation.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="tugasakhir_sintetis_10menit")
    parser.add_argument("--output", type=Path, default=Path("hasil_ews_keras"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-windows", type=int, default=168)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.resolve() == root or any(args.output.resolve().is_relative_to((root / name).resolve())
            for name in ("hasil_ews", "hasil_ews_pytorch", "hasil_ews_pytorch_device", "uji_model", "hasil_salin_sintetis", "audit_timestamp")):
        parser.error("Gunakan folder baru khusus hasil Keras.")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output sudah berisi hasil. Pakai --resume atau --output baru.")
    if args.min_train_windows < 24:
        parser.error("Minimal train windows harus >=24.")
    config = Config(epochs=args.epochs, patience=args.patience, hidden_size=args.hidden_size, seed=args.seed)
    config.validate()
    if keras.backend.backend() != "tensorflow":
        parser.error("Versi ini memakai backend TensorFlow. Set KERAS_BACKEND=tensorflow.")
    tf.config.experimental.enable_op_determinism()
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.json"
    snapshot = args.output / "hourly_snapshot.pkl"
    if args.resume:
        previous = json.loads(run_path.read_text())
        expected = {"config": asdict(config), "database": args.database, "min_train_windows": args.min_train_windows,
                    "keras_version": str(keras.__version__), "tensorflow_version": str(tf.__version__)}
        if any(previous.get(k) != v for k, v in expected.items()):
            parser.error("Konfigurasi/database/versi berbeda dari eksperimen sebelumnya.")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != previous["snapshot_sha256"]:
            parser.error("Snapshot berubah; gunakan folder baru.")
        hourly = pd.read_pickle(snapshot)
        run_info = previous
    else:
        hourly, audit, source = load_synthetic(args.database)
        pd.to_pickle(hourly, snapshot)
        audit.to_csv(args.output / "audit_data.csv", index=False)
        run_info = dict(source, config=asdict(config), min_train_windows=args.min_train_windows,
                        keras_version=str(keras.__version__), tensorflow_version=str(tf.__version__),
                        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                        cpu_threads=1, interval_model_hours=1, time_semantics="synthetic_not_actual_measurement")
    run_info["status"] = "running"
    write_json(run_path, run_info)
    manifest, ranking = train(hourly, args.output, config, args.min_train_windows, args.resume)
    report(args.output, manifest)
    outputs, audit = [], []
    for device, frame in hourly.items():
        try:
            future = forecast(frame, args.output)
            future["device_id"] = device
            outputs.append(future.reset_index())
            audit.append({"device_id": device, "result": "ok"})
        except ValueError as error:
            audit.append({"device_id": device, "result": str(error)})
    (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()).to_csv(args.output / "prediksi_24_jam.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output / "audit_prediksi.csv", index=False)
    run_info["status"] = "complete"
    write_json(run_path, run_info)
    if len(ranking):
        print(ranking.loc[ranking.is_champion, ["sensor", "device_id", "validation_macro_mae", "test_macro_mae", "test_baseline_macro_mae"]].to_string(index=False))
    print("Tersimpan:", args.output.resolve())


if __name__ == "__main__":
    main()
