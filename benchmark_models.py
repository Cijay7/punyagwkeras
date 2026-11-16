"""Uji historis reproducible; input snapshot hourly, tidak menulis model produksi."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from ews_gambut import train_parameters, supervised


def benchmark(snapshot="uji_model/hourly_snapshot.pkl", output="uji_model"):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_pickle(snapshot)  # hanya snapshot lokal tepercaya
    results, per_device, selections = [], [], []
    for fold, fraction in (("earlier", 0.8), ("latest", 1.0)):
        # Gunakan waktu asal supervised agar perangkat dengan gap panjang tidak mendominasi cutoff.
        origins = sorted(set(t for frame in hourly.values() for t in supervised(frame)[0].index))
        cutoff = origins[min(int(len(origins) * fraction), len(origins)) - 1] + pd.Timedelta(hours=24)
        frames = {device: frame.loc[:cutoff] for device, frame in hourly.items()}
        print(f"FOLD {fold}: cutoff={cutoff}", flush=True)
        for strategy in ("legacy", "improved", "conservative"):
            bundles, metric = train_parameters(frames, strategy=strategy)
            metric["fold"], metric["strategy"] = fold, strategy
            results.append(metric)
            for sensor, bundle in bundles.items():
                meta = bundle["metadata"]
                selections.append({"fold": fold, "strategy": strategy, "sensor": sensor,
                                   "metadata": meta})
                for row in meta["test_per_device"]:
                    per_device.append(dict(row, fold=fold, strategy=strategy))
            if fold == "latest":
                joblib.dump(bundles, folder / f"bundles_{strategy}.joblib")
            print(f"  {strategy} selesai", flush=True)
    scores = pd.concat(results, ignore_index=True)
    scores.to_csv(folder / "evaluasi_kandidat.csv", index=False)
    pd.DataFrame(per_device).to_csv(folder / "evaluasi_per_device.csv", index=False)
    (folder / "pemilihan_validation.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    return summarize(folder)


def summarize(folder):
    folder = Path(folder)
    scores = pd.read_csv(folder / "evaluasi_kandidat.csv")
    summary = scores.groupby(["fold", "sensor", "strategy"])[["mae", "mae_baseline"]].mean().reset_index()
    comparison = summary.pivot(index=["fold", "sensor"], columns="strategy", values="mae").reset_index()
    comparison = comparison.merge(summary.loc[summary.strategy.eq("legacy"), ["fold", "sensor", "mae_baseline"]], on=["fold", "sensor"])
    for strategy in ("improved", "conservative"):
        comparison[f"{strategy}_vs_legacy_pct"] = np.where(comparison.legacy > 0, 100 * (1 - comparison[strategy] / comparison.legacy), np.nan)
        comparison[f"{strategy}_vs_baseline_pct"] = np.where(comparison.mae_baseline > 0, 100 * (1 - comparison[strategy] / comparison.mae_baseline), np.nan)
    comparison.to_csv(folder / "perbandingan_model.csv", index=False)
    latest = comparison.loc[comparison.fold.eq("latest")]
    lines = ["# Hasil uji historis model EWS gambut", "",
             "Perbaikan diterapkan pada SM dan ST. Parameter lainnya memakai nilai terakhir.", "",
             "MAE rata-rata horizon 1-24 jam pada origin test yang sama, dalam satuan parameter.", "",
             "| Parameter | Model lama | Model diterapkan | Baseline | Penurunan MAE dari lama |",
             "|---|---:|---:|---:|---:|"]
    for row in latest.itertuples():
        lines.append(f"| {row.sensor} | {row.legacy:.6f} | {row.conservative:.6f} | {row.mae_baseline:.6f} | {row.conservative_vs_legacy_pct:.2f}% |")
    lines += ["", "## Batas kesimpulan", "",
              "- Kebijakan SM/ST dipilih setelah membaca benchmark historis. Angka ini bukan evaluasi independen pada data baru.",
              "- SM hanya bervariasi pada satu device yang berkontribusi ke test terbaru. MAE gabungan rendah karena device SM lain konstan.",
              "- ST membaik secara gabungan, tetapi beberapa device memburuk. Lihat evaluasi_per_device.csv.",
              "- Pada test terbaru, kandidat semua parameter menaikkan MAE RR 12,64% dan SEC 5,51%; keduanya tidak diterapkan. Perbaikan WL hanya 0,80%, sehingga tetap baseline.",
              "- Sampel test saling tumpang tindih. Tidak dihitung interval kepercayaan atau akurasi kebakaran.",
              "- Histori dengan jendela training lengkap berakhir 25 Mei 2026. Prediksi yang dihasilkan tetap historis/data lama.", "",
              "## Reproduksi", "",
              "`python benchmark_models.py --snapshot uji_model/hourly_snapshot.pkl`", "",
              "Snapshot berasal dari pembacaan MySQL read-only, dibersihkan dengan prepare_device. Lag dan target disusun per device sebelum digabung.",
              "Kandidat dipilih menggunakan validation, dengan gap target 24 jam sebelum validation dan test. Dua cutoff historis diuji tanpa tuning kandidat setelah melihat test.",
              "Pemilihan metode per blok horizon memakai MAE validation dan syarat perbaikan 2%. Kandidat pohon menggunakan seed 42.",
              "Rincian batas waktu, jumlah jendela, metode dan MAE validation tersimpan di pemilihan_validation.json.", "",
              "## Input yang bervariasi", "",
              "MAE di bawah hanya memakai origin dengan 24 jam input yang tidak konstan; penyebutnya bukan semua origin test.", "",
              "| Parameter | Origin bervariasi | Total origin | MAE input bervariasi | Baseline input bervariasi |",
              "|---|---:|---:|---:|---:|"]
    active = scores.loc[scores.fold.eq("latest") & scores.strategy.eq("conservative")].groupby("sensor").agg(
        origins=("nonconstant_input_origins", "first"), total=("test_origins", "first"),
        mae=("mae_nonconstant_input", "mean"), baseline=("mae_baseline_nonconstant_input", "mean"))
    for sensor, row in active.iterrows():
        lines.append(f"| {sensor} | {int(row.origins)} | {int(row.total)} | {row.mae:.6f} | {row.baseline:.6f} |")
    lines += ["", "Referensi implementasi: [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html), [evaluasi waktu](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)."]
    (folder / "hasil_uji.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    for ax, sensor in zip(axes.flat, ("SM", "ST", "ph", "SEC", "WL", "RR")):
        part = scores.loc[scores.fold.eq("latest") & scores.sensor.eq(sensor)]
        old = part.loc[part.strategy.eq("legacy")].sort_values("horizon_jam")
        new = part.loc[part.strategy.eq("conservative")].sort_values("horizon_jam")
        ax.plot(old.horizon_jam, old.mae, label="Model lama", color="#b45309", linestyle="--")
        ax.plot(new.horizon_jam, new.mae, label="Model diterapkan", color="#1565c0")
        ax.plot(old.horizon_jam, old.mae_baseline, label="Nilai terakhir", color="#555555", linestyle=":")
        ax.set(title=sensor, xlabel="Horizon (jam)", ylabel="MAE (satuan parameter)", ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Uji historis terbaru | MAE sensor, bukan akurasi kebakaran")
    fig.savefig(folder / "perbandingan_mae.png", dpi=140)
    plt.close(fig)
    print(comparison.round(6).to_string(index=False), flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="uji_model/hourly_snapshot.pkl")
    parser.add_argument("--output", default="uji_model")
    args = parser.parse_args()
    benchmark(args.snapshot, args.output)
