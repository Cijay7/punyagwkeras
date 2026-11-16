"""Eksperimen GRU per parameter; seluruh keluaran terpisah dari hasil_ews/."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ews_gambut import (FEATURES, HORIZON, LAGS, classify_rules, load_mysql,
                        parameter_bounds, predictors, prepare_device, supervised,
                        train_parameters, validate_rules)


@dataclass
class TrainingConfig:
    hidden_size: int = 16
    epochs: int = 60
    patience: int = 8
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 42

    def validate(self):
        if min(self.hidden_size, self.epochs, self.patience, self.batch_size) < 1:
            raise ValueError("Ukuran model, epochs, patience dan batch harus positif.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate harus positif dan finite.")


class ParameterGRU(nn.Module):
    """Urutan 24 jam -> perubahan untuk 24 jam berikutnya secara langsung."""
    def __init__(self, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, HORIZON)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sequence):
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


def parameter_samples(hourly_by_device, sensor):
    for device, frame in hourly_by_device.items():
        if (not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique or
                not frame.index.is_monotonic_increasing or
                (len(frame) > 1 and not (frame.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all())):
            raise ValueError(f"{device}: snapshot harus memiliki indeks per jam berurutan; gap harus berupa NaN.")
    samples = {device: supervised(frame[[sensor]]) for device, frame in hourly_by_device.items()}
    samples = {device: pair for device, pair in samples.items() if len(pair[0])}
    if not samples:
        raise ValueError(f"{sensor}: tidak ada jendela lengkap.")
    x = pd.concat({d: pair[0] for d, pair in samples.items()}, names=["device_id", "date_time"])
    y = pd.concat({d: pair[1] for d, pair in samples.items()}, names=["device_id", "date_time"])
    return x, y


def temporal_split(x):
    times = x.index.get_level_values("date_time")
    origins = times.unique().sort_values()
    if len(origins) < 168:
        raise ValueError("Perlu minimal 168 waktu asal dengan jendela lengkap per parameter.")
    validation_time, test_time = origins[int(len(origins) * .6)], origins[int(len(origins) * .8)]
    masks = {
        "train": times + pd.Timedelta(hours=HORIZON) < validation_time,
        "validation": (times >= validation_time) & (times + pd.Timedelta(hours=HORIZON) < test_time),
        "test": times >= test_time,
        "before_test": times + pd.Timedelta(hours=HORIZON) < test_time,
    }
    if min(int(masks[k].sum()) for k in ("train", "validation", "test")) < 24:
        raise ValueError("Blok train/validation/test setelah gap 24 jam terlalu pendek.")
    return masks, {"validation_start": str(validation_time), "test_start": str(test_time)}


def sequence_input(x, sensor, scale):
    # Lag23 adalah titik terlama; lag0 titik terakhir. Tidak menghubungkan device.
    lag = x[[f"{sensor}_lag{i}" for i in range(LAGS - 1, -1, -1)]].to_numpy(dtype=np.float32)
    level = lag[:, [-1]]
    times = x.index.get_level_values("date_time") if isinstance(x.index, pd.MultiIndex) else x.index
    hours = np.asarray(times.hour)[:, None] - np.arange(LAGS - 1, -1, -1)[None, :]
    relative = (lag - level) / scale
    sequence = np.stack([relative, np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], axis=-1)
    if not np.isfinite(sequence).all():
        raise ValueError("Input GRU tidak lengkap atau tidak finite.")
    return torch.from_numpy(sequence.astype(np.float32)), level.astype(float)


def predict_gru(model, x, sensor, scale):
    sequence, level = sequence_input(x, sensor, scale)
    model.eval()
    with torch.inference_mode():
        delta = torch.cat([model(batch) for batch in sequence.split(512)]).numpy()
    result = parameter_bounds(level + delta * scale, sensor)
    if result.shape != (len(x), HORIZON) or not np.isfinite(result).all():
        raise ValueError("Keluaran GRU tidak valid.")
    return result


def fit_gru(x, y, sensor, config, validation=None, fixed_epochs=None):
    """Scaler hanya belajar dari x/y bagian fit, tidak dari validation atau test."""
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    level = x[f"{sensor}_lag0"].to_numpy()[:, None]
    delta = y.to_numpy() - level
    if not np.isfinite(delta).all():
        raise ValueError("Target training tidak lengkap atau tidak finite.")
    scale = max(float(np.std(delta)), 1e-6)
    sequence, _ = sequence_input(x, sensor, scale)
    targets = torch.tensor(delta / scale, dtype=torch.float32)
    loader = DataLoader(TensorDataset(sequence, targets), batch_size=config.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(config.seed))
    model = ParameterGRU(config.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    history, best_state, best_epoch, best_loss, waiting = [], None, 0, float("inf"), 0
    for epoch in range(1, (fixed_epochs if fixed_epochs is not None else config.epochs) + 1):
        model.train()
        losses = []
        for inputs, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()) * len(inputs))
        row = {"epoch": epoch, "train_l1_scaled": sum(losses) / len(x)}
        if validation is not None:
            vx, vy = validation
            score = float(np.abs(predict_gru(model, vx, sensor, scale) - vy.to_numpy()).mean())
            row["validation_mae"] = score
            if score < best_loss - 1e-8:
                best_loss, best_epoch, waiting = score, epoch, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                waiting += 1
        history.append(row)
        if validation is not None and waiting >= config.patience:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = int(fixed_epochs if fixed_epochs is not None else config.epochs)
    return model, scale, best_epoch, history


def save_checkpoint(path, model, scale, metadata, config):
    torch.save({"state_dict": model.state_dict(), "scale": float(scale),
                "metadata": metadata, "config": asdict(config)}, path)


def load_checkpoint(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    meta = bundle["metadata"]
    if (meta["sensor"] not in FEATURES or meta["lag_hours"] != LAGS or
            meta["horizon_hours"] != HORIZON or meta["framework"] != "pytorch"):
        raise ValueError("Checkpoint bukan model parameter PyTorch yang sesuai.")
    if not np.isfinite(bundle["scale"]) or bundle["scale"] <= 0:
        raise ValueError("Skala checkpoint tidak valid.")
    model = ParameterGRU(bundle["config"]["hidden_size"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle["scale"], meta


def selected_prediction(raw, baseline, methods):
    return np.where(np.asarray(methods)[None, :] == "GRU", raw, baseline)


def plot_comparison(output, metric, classical_scores=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for ax, sensor in zip(axes.flat, FEATURES):
        part = metric.loc[metric.sensor.eq(sensor)].sort_values("horizon_jam")
        ax.plot(part.horizon_jam, part.mae_gru, label="GRU murni", color="#1565c0")
        ax.plot(part.horizon_jam, part.mae_selected, label="GRU / baseline (validation)", color="#7e22ce", linestyle="-.")
        ax.plot(part.horizon_jam, part.mae_baseline, label="Nilai terakhir", color="#555555", linestyle=":")
        if classical_scores is not None:
            old = classical_scores.loc[classical_scores.sensor.eq(sensor)].sort_values("horizon_jam")
            ax.plot(old.horizon_jam, old.mae, label="Model klasik saat ini", color="#b45309", linestyle="--")
        ax.set(title=sensor, xlabel="Horizon (jam)", ylabel="MAE (satuan parameter)", ylim=(0, None))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Perbandingan historis PyTorch GRU | split dan origin test yang sama")
    fig.savefig(Path(output) / "perbandingan_mae.png", dpi=140)
    plt.close(fig)


def train_pytorch(hourly_by_device, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, device_metrics, predictions, all_metadata, bundles = [], [], [], {}, {}
    for sensor in FEATURES:
        print(f"Training GRU {sensor}...", flush=True)
        x, y = parameter_samples(hourly_by_device, sensor)
        masks, split_info = temporal_split(x)
        model, scale, epochs, history = fit_gru(x.loc[masks["train"]], y.loc[masks["train"]], sensor,
                                              config, (x.loc[masks["validation"]], y.loc[masks["validation"]]))
        vx, vy = x.loc[masks["validation"]], y.loc[masks["validation"]].to_numpy()
        val_pred = predict_gru(model, vx, sensor, scale)
        val_base = np.repeat(vx[f"{sensor}_lag0"].to_numpy()[:, None], HORIZON, axis=1)
        methods = []
        for start, end in ((0, 6), (6, 12), (12, 24)):
            better = np.abs(val_pred[:, start:end] - vy[:, start:end]).mean() < .98 * np.abs(val_base[:, start:end] - vy[:, start:end]).mean()
            methods.extend(["GRU" if better else "Nilai terakhir"] * (end - start))
        # Epoch dipilih di validation. Refit sebelum test tanpa early stopping pada test.
        test_model, test_scale, _, _ = fit_gru(x.loc[masks["before_test"]], y.loc[masks["before_test"]],
                                              sensor, config, fixed_epochs=epochs)
        tx, actual = x.loc[masks["test"]], y.loc[masks["test"]].to_numpy()
        raw = predict_gru(test_model, tx, sensor, test_scale)
        baseline = np.repeat(tx[f"{sensor}_lag0"].to_numpy()[:, None], HORIZON, axis=1)
        selected = selected_prediction(raw, baseline, methods)
        varying = np.ptp(tx[[f"{sensor}_lag{i}" for i in range(LAGS)]].to_numpy(), axis=1) > 0
        for h in range(HORIZON):
            metrics.append({"sensor": sensor, "horizon_jam": h + 1, "metode_terpilih": methods[h],
                            "mae_gru": float(np.abs(raw[:, h] - actual[:, h]).mean()),
                            "mae_selected": float(np.abs(selected[:, h] - actual[:, h]).mean()),
                            "mae_baseline": float(np.abs(baseline[:, h] - actual[:, h]).mean()),
                            "test_origins": len(tx), "nonconstant_input_origins": int(varying.sum()),
                            "mae_gru_nonconstant": float(np.abs(raw[varying, h] - actual[varying, h]).mean()) if varying.any() else None})
        devices = tx.index.get_level_values("device_id")
        for device in devices.unique():
            mask = devices == device
            device_metrics.append({"sensor": sensor, "device_id": str(device), "test_origins": int(mask.sum()),
                                   "mae_gru": float(np.abs(raw[mask] - actual[mask]).mean()),
                                   "mae_selected": float(np.abs(selected[mask] - actual[mask]).mean()),
                                   "mae_baseline": float(np.abs(baseline[mask] - actual[mask]).mean())})
        records = tx.index.to_frame(index=False).loc[tx.index.to_frame(index=False).index.repeat(HORIZON)].reset_index(drop=True)
        records["sensor"], records["horizon_jam"] = sensor, np.tile(np.arange(1, HORIZON + 1), len(tx))
        for name, values in (("actual", actual), ("gru", raw), ("selected", selected), ("baseline", baseline)):
            records[name] = values.reshape(-1)
        predictions.append(records)
        times = x.index.get_level_values("date_time")
        meta = {"sensor": sensor, "framework": "pytorch", "architecture": "GRU",
                "training_scope": "parameter", "training_devices": [str(d) for d in x.index.get_level_values("device_id").unique()],
                "lag_hours": LAGS, "horizon_hours": HORIZON, "best_epoch": epochs,
                "method_per_horizon": methods, **split_info,
                "train_windows": int(masks["train"].sum()), "validation_windows": int(masks["validation"].sum()),
                "test_windows": int(masks["test"].sum()), "test_origin_end": str(tx.index.get_level_values("date_time").max()),
                "validation_mae_gru": float(np.abs(val_pred - vy).mean()),
                "validation_mae_baseline": float(np.abs(val_base - vy).mean())}
        evaluation_meta = dict(meta, trained_until=str((times[masks["before_test"]] + pd.Timedelta(hours=HORIZON)).max()))
        save_checkpoint(output / f"evaluasi_parameter_{sensor}.pt", test_model, test_scale, evaluation_meta, config)
        final_model, final_scale, _, _ = fit_gru(x, y, sensor, config, fixed_epochs=epochs)
        meta["trained_until"] = str((times + pd.Timedelta(hours=HORIZON)).max())
        save_checkpoint(output / f"model_parameter_{sensor}.pt", final_model, final_scale, meta, config)
        pd.DataFrame(history).to_csv(output / f"training_{sensor}.csv", index=False)
        bundles[sensor] = (final_model, final_scale, meta)
        all_metadata[sensor] = meta
        print(f"  epoch={epochs}; test MAE GRU={np.abs(raw-actual).mean():.6f}; baseline={np.abs(baseline-actual).mean():.6f}", flush=True)
    metric = pd.DataFrame(metrics)
    metric.to_csv(output / "evaluasi_mae.csv", index=False)
    pd.DataFrame(device_metrics).to_csv(output / "evaluasi_per_device.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "prediksi_test.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")
    return bundles, metric


def forecast_device(hourly, bundles, config, as_of):
    if hourly.empty:
        raise ValueError("Tidak ada data per jam.")
    origin = hourly.index[-1]
    future = pd.DataFrame(index=pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON, freq="h"))
    future.index.name = "date_time"
    for sensor in FEATURES:
        model, scale, meta = bundles[sensor]
        if pd.Timestamp(meta["trained_until"]) > origin:
            raise ValueError("Model dilatih melewati waktu input perangkat.")
        last_x = predictors(hourly[[sensor]]).iloc[[-1]]
        raw = predict_gru(model, last_x, sensor, scale)[0]
        baseline = np.full(HORIZON, hourly.iloc[-1][sensor])
        future[sensor] = raw
        future[f"{sensor}_selected"] = selected_prediction(raw[None, :], baseline[None, :], meta["method_per_horizon"])[0]
    future["forecast_origin"] = origin
    future["data_lama"] = bool(pd.Timestamp(as_of) - origin > pd.Timedelta(hours=2))
    # Status utama memakai prediksi GRU murni, kolom *_selected untuk pembanding.
    return future.join(classify_rules(future, config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot hourly lokal; tanpa opsi ini membaca MySQL")
    parser.add_argument("--output", type=Path, default=Path("hasil_ews_pytorch"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--as-of", help="Batas baca MySQL dan pemeriksaan kesegaran")
    parser.add_argument("--skip-classical", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protected = [root / "hasil_ews", root / "uji_model"]
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in protected):
        parser.error("Gunakan folder output terpisah dari hasil_ews dan uji_model.")
    config = TrainingConfig(hidden_size=args.hidden_size, epochs=args.epochs, patience=args.patience, seed=args.seed)
    config.validate()
    torch.set_num_threads(1)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    rules = json.loads((root / "aturan_risiko.json").read_text(encoding="utf-8"))
    validate_rules(rules)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        hourly = pd.read_pickle(args.snapshot)  # snapshot lokal tepercaya
        source = {"snapshot": str(args.snapshot.resolve()), "sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest()}
    else:
        raw, coverage = load_mysql(as_of=as_of)
        hourly = {str(d): prepare_device(raw, d)[0] for d in raw.device_id.dropna().unique()}
        pd.to_pickle(hourly, args.output / "hourly_snapshot.pkl")
        coverage.to_csv(args.output / "cakupan_data.csv", index=False)
        source = {"source": "MySQL sensor_data read-only", "as_of": str(as_of)}
    bundles, metric = train_pytorch(hourly, args.output, config)
    summary = metric.groupby("sensor")[["mae_gru", "mae_selected", "mae_baseline"]].mean()
    scores = None
    if not args.skip_classical:
        print("Menghitung ulang pembanding klasik pada snapshot dan split yang sama...", flush=True)
        classical, scores = train_parameters(hourly)
        for sensor, bundle in classical.items():
            for key in ("validation_start", "test_start", "test_windows"):
                if bundle["metadata"][key] != bundles[sensor][2][key]:
                    raise ValueError("Split pembanding klasik tidak sama.")
        summary["mae_classical"] = scores.groupby("sensor").mae.mean()
        scores.to_csv(args.output / "evaluasi_classical.csv", index=False)
    summary["penurunan_gru_vs_baseline_pct"] = 100 * (1 - summary.mae_gru / summary.mae_baseline.replace(0, np.nan))
    summary.to_csv(args.output / "perbandingan_model.csv")
    plot_comparison(args.output, metric, scores)
    source.update(config=asdict(config), torch_version=str(torch.__version__), cpu_threads=1,
                  note="Eksperimen historis; test belum merupakan validasi pada data masa depan baru.")
    (args.output / "run.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    forecasts, audits = [], []
    for device, frame in hourly.items():
        try:
            future = forecast_device(frame, bundles, rules, as_of)
            future["device_id"] = device
            forecasts.append(future.reset_index())
            audits.append({"device_id": device, "result": "ok"})
        except ValueError as error:
            audits.append({"device_id": device, "result": str(error)})
    (pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()).to_csv(args.output / "prediksi_24_jam.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output / "audit_perangkat.csv", index=False)
    print(summary.round(6).to_string())
    print(f"Hasil eksperimen tersimpan di {args.output.resolve()}")


if __name__ == "__main__":
    main()
