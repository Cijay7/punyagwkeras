"""Train GRU per device-parameter, pilih satu pemenang per parameter di validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ews_gambut import FEATURES, HORIZON, load_mysql, prepare_device, validate_rules
from ews_gambut_pytorch import (TrainingConfig, fit_gru, forecast_device,
                                load_checkpoint, parameter_samples, predict_gru,
                                save_checkpoint, temporal_split)


def device_folder(device):
    value = str(device)
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
        raise ValueError("ID perangkat hanya boleh berisi huruf, angka, - dan _.")
    return value


def fit_fingerprint(x, y):
    digest = hashlib.sha256()
    for frame in (x, y):
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update("|".join(frame.columns).encode("utf-8"))
    return digest.hexdigest()


def save_atomic(path, model, scale, metadata, config):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    save_checkpoint(temporary, model, scale, metadata, config)
    temporary.replace(path)


def write_progress(output, candidates, audit, horizons, manifest):
    for name, rows in (("ranking_model.csv", candidates), ("audit_device_parameter.csv", audit),
                       ("evaluasi_horizon.csv", horizons)):
        path = output / name
        temporary = path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(path)
    path = output / "model_terbaik.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def cached_fit(path, x, y, sensor, device, config, split_info, resume,
               validation=None, fixed_epochs=None, history_path=None):
    """Reuse hanya tahap fit yang lengkap dan cocok dengan data/konfigurasi."""
    path = Path(path)
    if not resume or not path.exists():
        return None
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        model, scale, meta = load_checkpoint(path)
        expected_until = str((x.index.get_level_values("date_time") + pd.Timedelta(hours=HORIZON)).max())
        expected_scale = max(float(np.std(y.to_numpy() - x[f"{sensor}_lag0"].to_numpy()[:, None])), 1e-6)
        if (bundle["config"] != asdict(config) or meta.get("device_id") != device or
                meta.get("training_devices") != [device] or meta.get("training_scope") != "device_parameter" or
                meta.get("sensor") != sensor or meta.get("trained_until") != expected_until or
                any(meta.get(key) != value for key, value in split_info.items()) or
                not np.isclose(scale, expected_scale, rtol=1e-10, atol=1e-12)):
            raise ValueError("Checkpoint berbeda dari konfigurasi atau data fit.")
        fingerprint = fit_fingerprint(x, y)
        if meta.get("fit_fingerprint", fingerprint) != fingerprint:
            raise ValueError("Isi data fit berubah.")
        epochs = int(meta["best_epoch"])
        if not 1 <= epochs <= config.epochs or (fixed_epochs is not None and epochs != fixed_epochs):
            raise ValueError("Jumlah epoch checkpoint berbeda.")
        history = []
        if validation is not None:
            history = pd.read_csv(history_path).to_dict("records")
            vx, vy = validation
            score = float(np.abs(predict_gru(model, vx, sensor, scale) - vy.to_numpy()).mean())
            saved_loss = next(row["validation_mae"] for row in history if row["epoch"] == epochs)
            if not np.isclose(score, saved_loss, rtol=1e-7, atol=1e-7):
                raise ValueError("Validation checkpoint tidak cocok dengan histori.")
        # Migrasi checkpoint sebelum crash: cek metadata/skala dan replay validation
        # pada snapshot lama yang sama sebelum menambahkan fingerprint tahap fit.
        if "fit_fingerprint" not in meta:
            meta["fit_fingerprint"] = fingerprint
            save_atomic(path, model, scale, meta, config)
        print(f"    resume {path.name}", flush=True)
        return model, scale, epochs, history
    except Exception as error:
        print(f"    checkpoint {path.name} diulang: {type(error).__name__}", flush=True)
        return None


def reference_devices(x, y, train, validation, min_windows=24):
    """Eligibility pembanding ditentukan dari train/validation saja, bukan test."""
    devices = x.index.get_level_values("device_id")
    references = []
    for device in devices.unique():
        source = devices == device
        mask = source & train
        if mask.sum() < min_windows or (source & validation).sum() < min_windows:
            continue
        sensor = x.columns[0].rsplit("_lag", 1)[0]
        delta = y.loc[mask].to_numpy() - x.loc[mask, f"{sensor}_lag0"].to_numpy()[:, None]
        if np.abs(delta).mean() > 1e-8:
            references.append(str(device))
    return references


def common_score(x, y, predicted, sensor, references):
    """MAE macro: bobot setiap device pembanding sama, pada baris yang sama."""
    devices = x.index.get_level_values("device_id")
    actual = y.to_numpy()
    baseline = np.repeat(x[f"{sensor}_lag0"].to_numpy()[:, None], HORIZON, axis=1)
    rows = []
    for device in references:
        mask = devices == device
        if not mask.any():
            continue
        rows.append({"device_id": device, "origins": int(mask.sum()),
                     "mae": float(np.abs(predicted[mask] - actual[mask]).mean()),
                     "mae_baseline": float(np.abs(baseline[mask] - actual[mask]).mean())})
    if not rows:
        raise ValueError("Tidak ada device pembanding pada blok ini.")
    mae = float(np.mean([row["mae"] for row in rows]))
    baseline_mae = float(np.mean([row["mae_baseline"] for row in rows]))
    return {"macro_mae": mae, "macro_mae_baseline": baseline_mae,
            "skill_pct": float(100 * (1 - mae / baseline_mae)) if baseline_mae > 1e-8 else None,
            "devices": rows}


def choose_winner(candidates):
    """Fungsi seleksi sengaja tidak membaca kolom test."""
    eligible = [row for row in candidates if row["eligible_champion"]]
    return min(eligible, key=lambda row: (row["validation_macro_mae"], row["device_id"])) if eligible else None


def train_device_parameters(hourly, output, config, min_train_windows=168, resume=False):
    config.validate()
    if min_train_windows < 24:
        raise ValueError("Minimal jendela training tidak boleh kurang dari 24.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit, all_candidates, horizon_scores, manifest = [], [], [], {}
    for sensor in FEATURES:
        print(f"\nPARAMETER {sensor}", flush=True)
        try:
            x, y = parameter_samples(hourly, sensor)
            masks, split_info = temporal_split(x)
        except ValueError as error:
            for device in hourly:
                audit.append({"sensor": sensor, "device_id": str(device), "result": str(error)})
            manifest[sensor] = {"status": "unavailable", "reason": str(error)}
            write_progress(output, all_candidates, audit, horizon_scores, manifest)
            continue
        devices = x.index.get_level_values("device_id")
        references = reference_devices(x, y, masks["train"], masks["validation"])
        comparison = devices.isin(references)
        validation_mask, test_mask = masks["validation"] & comparison, masks["test"] & comparison
        candidates = []
        for device in hourly:
            device = str(device)
            source = devices == device
            train_mask, val_mask = masks["train"] & source, masks["validation"] & source
            row = {"sensor": sensor, "device_id": device, "train_windows": int(train_mask.sum()),
                   "validation_windows": int(val_mask.sum()), "test_windows": int((masks["test"] & source).sum())}
            if train_mask.sum() < min_train_windows or val_mask.sum() < 24:
                audit.append(dict(row, result="Histori train/validation perangkat tidak cukup"))
                continue
            folder = output / "devices" / device_folder(device)
            folder.mkdir(parents=True, exist_ok=True)
            source_delta = y.loc[train_mask].to_numpy() - x.loc[train_mask, f"{sensor}_lag0"].to_numpy()[:, None]
            constant = bool(np.abs(source_delta).mean() <= 1e-8)
            candidate_config = TrainingConfig(**asdict(config))
            # Sensor tanpa perubahan tetap memperoleh checkpoint, tetapi tidak ikut ranking.
            if constant:
                candidate_config.epochs = 1
            print(f"  {device}: train={train_mask.sum()}, validation={val_mask.sum()}" + (" (konstan)" if constant else ""), flush=True)
            validation = (x.loc[val_mask], y.loc[val_mask])
            cached = cached_fit(folder / f"validation_parameter_{sensor}.pt", x.loc[train_mask], y.loc[train_mask],
                                sensor, device, candidate_config, split_info, resume, validation=validation,
                                history_path=folder / f"training_{sensor}.csv")
            model, scale, epochs, history = cached if cached is not None else fit_gru(
                x.loc[train_mask], y.loc[train_mask], sensor, candidate_config, validation)
            own_prediction = predict_gru(model, x.loc[val_mask], sensor, scale)
            val_score = (common_score(x.loc[validation_mask], y.loc[validation_mask],
                                     predict_gru(model, x.loc[validation_mask], sensor, scale), sensor, references)
                         if validation_mask.any() else None)
            eligible = not constant and val_score is not None
            row.update(eligible_champion=bool(eligible), constant_train=constant, best_epoch=epochs,
                       validation_own_mae=float(np.abs(own_prediction - y.loc[val_mask].to_numpy()).mean()),
                       validation_macro_mae=val_score["macro_mae"] if val_score else None,
                       validation_baseline_macro_mae=val_score["macro_mae_baseline"] if val_score else None,
                       validation_skill_pct=val_score["skill_pct"] if val_score else None)
            times = x.index.get_level_values("date_time")
            meta = {"sensor": sensor, "device_id": device, "training_devices": [device],
                    "training_scope": "device_parameter", "framework": "pytorch", "architecture": "GRU",
                    "lag_hours": 24, "horizon_hours": HORIZON, "best_epoch": epochs,
                    "method_per_horizon": ["GRU"] * HORIZON, "constant_train": constant,
                    "reference_devices": references, **split_info,
                    "train_windows": int(train_mask.sum()), "validation_windows": int(val_mask.sum()),
                    "selection_metric": "validation macro MAE on the same nonconstant reference devices",
                    "selection_uses_test": False}
            validation_meta = dict(meta, trained_until=str((times[train_mask] + pd.Timedelta(hours=HORIZON)).max()),
                                   fit_fingerprint=fit_fingerprint(x.loc[train_mask], y.loc[train_mask]))
            save_atomic(folder / f"validation_parameter_{sensor}.pt", model, scale, validation_meta, candidate_config)
            pd.DataFrame(history).to_csv(folder / f"training_{sensor}.csv", index=False)
            candidates.append({**row, "metadata": meta, "config": candidate_config,
                               "folder": folder, "source_mask": source, "validation_details": val_score})
        # Kunci pemenang sebelum menjalankan inferensi test seluruh kandidat.
        winner = choose_winner(candidates)
        winner_device = winner["device_id"] if winner else None
        for candidate in candidates:
            device, source, folder = candidate["device_id"], candidate["source_mask"], candidate["folder"]
            meta, cfg, epochs = candidate["metadata"], candidate["config"], candidate["best_epoch"]
            before_test = masks["before_test"] & source
            cached = cached_fit(folder / f"evaluasi_parameter_{sensor}.pt", x.loc[before_test], y.loc[before_test],
                                sensor, device, cfg, split_info, resume, fixed_epochs=epochs)
            evaluated, evaluated_scale, _, _ = cached if cached is not None else fit_gru(
                x.loc[before_test], y.loc[before_test], sensor, cfg, fixed_epochs=epochs)
            times = x.index.get_level_values("date_time")
            evaluated_meta = dict(meta, trained_until=str((times[before_test] + pd.Timedelta(hours=HORIZON)).max()),
                                  fit_fingerprint=fit_fingerprint(x.loc[before_test], y.loc[before_test]))
            save_atomic(folder / f"evaluasi_parameter_{sensor}.pt", evaluated, evaluated_scale, evaluated_meta, cfg)
            test_details = None
            if test_mask.any():
                prediction = predict_gru(evaluated, x.loc[test_mask], sensor, evaluated_scale)
                test_details = common_score(x.loc[test_mask], y.loc[test_mask], prediction, sensor, references)
                actual = y.loc[test_mask].to_numpy()
                baseline = x.loc[test_mask, f"{sensor}_lag0"].to_numpy()[:, None]
                for h in range(HORIZON):
                    horizon_scores.append({"sensor": sensor, "source_device": device, "horizon_jam": h + 1,
                                           "mae": float(np.abs(prediction[:, h] - actual[:, h]).mean()),
                                           "mae_baseline": float(np.abs(baseline[:, 0] - actual[:, h]).mean()),
                                           "test_origins": int(test_mask.sum()), "is_champion": device == winner_device})
            own_test = masks["test"] & source
            own_test_mae = (float(np.abs(predict_gru(evaluated, x.loc[own_test], sensor, evaluated_scale) - y.loc[own_test].to_numpy()).mean())
                            if own_test.any() else None)
            # Model final hanya menerima histori device asal, termasuk saat fit ulang.
            cached = cached_fit(folder / f"model_parameter_{sensor}.pt", x.loc[source], y.loc[source],
                                sensor, device, cfg, split_info, resume, fixed_epochs=epochs)
            final, final_scale, _, _ = cached if cached is not None else fit_gru(
                x.loc[source], y.loc[source], sensor, cfg, fixed_epochs=epochs)
            final_meta = dict(meta, trained_until=str((times[source] + pd.Timedelta(hours=HORIZON)).max()),
                              fit_fingerprint=fit_fingerprint(x.loc[source], y.loc[source]))
            path = folder / f"model_parameter_{sensor}.pt"
            save_atomic(path, final, final_scale, final_meta, cfg)
            export = {key: candidate[key] for key in (
                "sensor", "device_id", "train_windows", "validation_windows", "test_windows",
                "eligible_champion", "constant_train", "best_epoch", "validation_own_mae",
                "validation_macro_mae", "validation_baseline_macro_mae", "validation_skill_pct")}
            export.update(is_champion=device == winner_device, test_own_mae=own_test_mae,
                          test_macro_mae=test_details["macro_mae"] if test_details else None,
                          test_baseline_macro_mae=test_details["macro_mae_baseline"] if test_details else None,
                          test_skill_pct=test_details["skill_pct"] if test_details else None,
                          model_path=str(path.relative_to(output)))
            all_candidates.append(export)
            audit.append(dict(export, result="Sensor konstan; dikecualikan dari ranking" if candidate["constant_train"] else "ok"))
            details = {"validation": candidate["validation_details"], "test": test_details,
                       "reference_devices": references, "split": split_info}
            (folder / f"evaluasi_{sensor}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
            if device == winner_device:
                champion_folder = output / "champions"
                champion_folder.mkdir(exist_ok=True)
                for prefix in ("model", "evaluasi", "validation"):
                    shutil.copy2(folder / f"{prefix}_parameter_{sensor}.pt", champion_folder / f"{prefix}_parameter_{sensor}.pt")
                manifest[sensor] = dict(export, status="selected", reference_devices=references,
                                        champion_path=f"champions/model_parameter_{sensor}.pt",
                                        selection_uses_test=False, **split_info)
                print(f"  PEMENANG {sensor}: {device}; validation={export['validation_macro_mae']:.6f}; test={export['test_macro_mae']}", flush=True)
        if winner is None:
            manifest[sensor] = {"status": "unavailable", "reason": "Tidak ada kandidat bervariasi dengan histori dan pembanding cukup"}
        # Simpan kemajuan tiap parameter agar eksperimen dapat diperiksa selama training.
        write_progress(output, all_candidates, audit, horizon_scores, manifest)
    ranking = pd.DataFrame(all_candidates)
    if len(ranking):
        ranking = ranking.sort_values(["sensor", "eligible_champion", "validation_macro_mae", "device_id"], ascending=[True, False, True, True])
        ranking.to_csv(output / "ranking_model.csv", index=False)
        ranking.loc[ranking.is_champion].to_csv(output / "model_terbaik.csv", index=False)
    else:
        pd.DataFrame(columns=["sensor", "device_id", "validation_macro_mae"]).to_csv(output / "model_terbaik.csv", index=False)
    return manifest, ranking


def plot_ranking(output):
    ranking = pd.read_csv(Path(output) / "ranking_model.csv")
    if ranking.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), layout="constrained")
    for ax, sensor in zip(axes.flat, FEATURES):
        part = ranking.loc[ranking.sensor.eq(sensor) & ranking.eligible_champion].sort_values("validation_macro_mae", ascending=False)
        if part.empty:
            ax.text(.5, .5, "Tidak ada kandidat layak", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.barh(part.device_id, part.validation_macro_mae,
                    color=["#1565c0" if win else "#777777" for win in part.is_champion])
            baseline = part.validation_baseline_macro_mae.iloc[0]
            ax.axvline(baseline, color="#b45309", linestyle="--", label="Baseline validation")
            ax.legend(fontsize=8)
        ax.set(title=sensor, xlabel="MAE validation macro (satuan parameter)", xlim=(0, None))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Ranking validation lintas device | biru = pemenang | sensor konstan dikecualikan")
    fig.savefig(Path(output) / "ranking_validation.png", dpi=140)
    plt.close(fig)


def forecast_champions(hourly, output, rules, as_of):
    output = Path(output)
    manifest = json.loads((output / "model_terbaik.json").read_text(encoding="utf-8"))
    missing = [s for s in FEATURES if manifest.get(s, {}).get("status") != "selected"]
    if missing:
        raise ValueError("Pemenang belum tersedia untuk: " + ", ".join(missing))
    bundles = {s: load_checkpoint(output / manifest[s]["champion_path"]) for s in FEATURES}
    return forecast_device(hourly, bundles, rules, as_of)


def write_report(output, manifest):
    output = Path(output)
    def fmt(value):
        return f"{value:.6f}" if value is not None else "-"
    lines = ["# Pemenang model per parameter", "",
             "Setiap kandidat dilatih hanya pada satu device. Pemenang dipilih dari MAE validation pada data pembanding lintas device yang sama, dengan bobot sama per device.", "",
             "Sensor dengan target training konstan tidak ikut ranking. Data test tidak dipakai memilih pemenang. Hasil tetap berupa eksperimen historis.", "",
             "| Parameter | Device asal model | MAE validation macro | MAE test macro | Baseline test |", "|---|---|---:|---:|---:|"]
    for sensor in FEATURES:
        row = manifest.get(sensor, {})
        if row.get("status") == "selected":
            lines.append(f"| {sensor} | {row['device_id']} | {fmt(row['validation_macro_mae'])} | {fmt(row['test_macro_mae'])} | {fmt(row['test_baseline_macro_mae'])} |")
        else:
            lines.append(f"| {sensor} | Belum tersedia | - | - | - |")
    lines += ["", "MAE mengikuti satuan masing-masing parameter. Nilai antarparameter tidak boleh dibandingkan langsung.",
              "Pemenang berarti terbaik di antara kandidat yang lolos syarat, bukan jaminan mengungguli baseline atau model gabungan sebelumnya.",
              "Model terbaik pada device asal saja belum tentu terbaik lintas device. Karena itu ranking utama memakai data validation pembanding yang sama; MAE device asal tetap disimpan terpisah.",
              "Jumlah origin per device dapat berbeda. MAE macro memberi bobot sama pada setiap device pembanding, bukan pada jumlah barisnya.",
              "Device pembanding ditentukan dari train dan validation yang cukup serta perubahan target training; daftar ini dikunci sebelum membaca test. Device tanpa origin test tidak berkontribusi ke skor test dan dicatat pada JSON evaluasi.",
              "Checkpoint validation, evaluasi dan final dipisah. Hanya checkpoint sebelum test yang menghitung MAE test; checkpoint final menerima seluruh histori device asal.",
              "Pemilihan epoch dan pemenang sama-sama memakai validation; konfirmasi pada data masa depan baru masih diperlukan. Origin yang tumpang tindih bukan percobaan independen.", "",
              "Lihat ranking_model.csv, audit_device_parameter.csv, evaluasi_horizon.csv dan devices/{device_id}/evaluasi_{sensor}.json untuk rincian."]
    (output / "hasil_pemilihan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=Path("hasil_ews_pytorch_device"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-windows", type=int, default=168)
    parser.add_argument("--as-of")
    parser.add_argument("--resume", action="store_true", help="Lanjutkan tahap fit dari checkpoint yang cocok")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if any(args.output.resolve().is_relative_to((root / name).resolve()) for name in ("hasil_ews", "uji_model", "hasil_ews_pytorch")):
        parser.error("Gunakan folder eksperimen device yang terpisah.")
    # Output baru mencegah checkpoint pemenang lama tertinggal ketika syarat data berubah.
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Folder output sudah berisi hasil; gunakan --output dengan folder baru.")
    config = TrainingConfig(hidden_size=args.hidden_size, epochs=args.epochs, patience=args.patience, seed=args.seed)
    config.validate()
    torch.set_num_threads(1)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    rules = json.loads((root / "aturan_risiko.json").read_text(encoding="utf-8"))
    validate_rules(rules)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        hourly = pd.read_pickle(args.snapshot)
        source = {"snapshot": str(args.snapshot.resolve()), "sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest()}
    elif args.resume and (args.output / "hourly_snapshot.pkl").exists():
        path = args.output / "hourly_snapshot.pkl"
        hourly = pd.read_pickle(path)
        source = {"snapshot": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        if args.resume:
            parser.error("Resume memerlukan --snapshot atau snapshot tersimpan dari pembacaan MySQL sebelumnya.")
        raw, coverage = load_mysql(as_of=as_of)
        hourly = {str(d): prepare_device(raw, d)[0] for d in raw.device_id.dropna().unique()}
        pd.to_pickle(hourly, args.output / "hourly_snapshot.pkl")
        coverage.to_csv(args.output / "cakupan_data.csv", index=False)
        source = {"source": "MySQL sensor_data read-only", "as_of": str(as_of)}
        path = args.output / "hourly_snapshot.pkl"
        source.update(snapshot=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source.update(config=asdict(config), min_train_windows=args.min_train_windows,
                  torch_version=str(torch.__version__), selection="common validation macro MAE", selection_uses_test=False)
    run_path = args.output / "run.json"
    if args.resume and run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key in ("sha256", "config", "min_train_windows", "torch_version"):
            if previous.get(key) != source.get(key):
                parser.error(f"Resume ditolak: {key} berbeda. Gunakan konfigurasi/snapshot asli atau folder baru.")
    source["status"] = "running"
    run_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    manifest, ranking = train_device_parameters(hourly, args.output, config, args.min_train_windows, resume=args.resume)
    write_report(args.output, manifest)
    plot_ranking(args.output)
    forecasts, audits = [], []
    for device, frame in hourly.items():
        try:
            future = forecast_champions(frame, args.output, rules, as_of)
            future["device_id"] = str(device)
            forecasts.append(future.reset_index())
            audits.append({"device_id": str(device), "result": "ok"})
        except ValueError as error:
            audits.append({"device_id": str(device), "result": str(error)})
    (pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()).to_csv(args.output / "prediksi_champions.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output / "audit_prediksi.csv", index=False)
    source["status"] = "complete"
    run_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    if len(ranking):
        print(ranking.loc[ranking.is_champion, ["sensor", "device_id", "validation_macro_mae", "test_macro_mae", "test_baseline_macro_mae"]].to_string(index=False))
    print("Tersimpan:", args.output.resolve())


if __name__ == "__main__":
    main()
