"""Susun notebook yang meneruskan koneksi MySQL awal, lalu validasi strukturnya."""
from pathlib import Path
import nbformat as nbf

path = Path("kecerdasancijay.ipynb")
backup = Path("kecerdasancijay.sebelum.ipynb")
if not backup.exists():
    backup.write_bytes(path.read_bytes())

cells = []
def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))
def code(source):
    cells.append(nbf.v4.new_code_cell(source))

md("""# EWS lahan gambut: prediksi 24 jam dan klasifikasi risiko

Notebook ini melanjutkan data MySQL `tugasakhir.sensor_data`. Keenam sensor
**SM, ST, ph, SEC, WL, RR** dilatih per parameter dari seluruh perangkat;
prediksi tetap ditampilkan per perangkat. Klasifikasi berbasis aturan
**Normal / Rawan / Bahaya** tersedia melalui variabel ambang yang dapat diedit.

**Kondisi awal:** belum ada label risiko; ambang sengaja kosong. Prediksi memakai
histori yang tersedia dan tidak boleh dianggap peringatan kondisi sekarang bila
telemetri sudah lama. Database tidak diubah. Notebook asli disimpan di
`kecerdasancijay.sebelum.ipynb`.
""")
md("""## 1. Persiapan
Jalankan notebook dari direktori proyek. Jika dependensi belum tersedia,
jalankan sel instalasi opsional di bawah. Koneksi mengikuti notebook awal;
untuk koneksi lain, isi environment variable `MYSQL_URL` di lingkungan lokal.
""")
code('# Opsional, cukup sekali:\n# %pip install -r requirements.txt')
code('''import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display
from ews_gambut import (FEATURES, load_mysql, prepare_device, run_all,
                        validate_rules, classify_rules)

plt.rcParams.update({"figure.figsize": (12, 5), "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.2})
pd.set_option("display.max_columns", 14)
HISTORY_START = "2026-01-01"
AS_OF = pd.Timestamp.now(tz="Asia/Makassar").tz_localize(None)
OUTPUT_DIR = "hasil_ews"
print("Batas baca:", AS_OF, "(waktu lokal; zona waktu database perlu dikonfirmasi)")''')
md("""## 2. Edit variabel ambang klasifikasi di sini

Ganti `None` dengan angka berdasarkan kalibrasi/pembimbing. Nilai awal kosong
agar tidak menghasilkan kategori risiko dari ambang yang ditebak.

| Sensor | Arah risiko | Rawan | Bahaya |
|---|---|---|---|
| SM (%) | Makin rendah | SM ≤ ambang rawan | SM ≤ ambang bahaya |
| ST (derajat) | Makin tinggi | ST ≥ ambang rawan | ST ≥ ambang bahaya |
| WL (cm) | Makin rendah | WL ≤ ambang rawan | WL ≤ ambang bahaya |

WL makin besar berarti air makin tinggi, sesuai keterangan pengguna.
Kategori akhir mengikuti pemicu terberat; Normal berarti tidak ada pemicu pada
aturan yang Anda isi. Kategori ini belum tervalidasi terhadap kejadian kebakaran.
ph dan SEC tetap diprediksi, tetapi belum menjadi pemicu risiko. RR belum menjadi
pemicu karena satuan dan jenis penghitungnya belum pasti.

Untuk mengaktifkan: isi **kedua** ambang pada sensor yang dipakai, catat sumbernya,
konfirmasikan satuannya, lalu ubah `AKTIFKAN_ATURAN = True`. Jika hanya satu
sensor diisi, klasifikasi hanya mencakup aturan sensor tersebut.
""")
code('''# ===== TEMPAT MENGEDIT AMBANG =====
SM_RAWAN = None
SM_BAHAYA = None
ST_RAWAN = None
ST_BAHAYA = None
WL_RAWAN = None
WL_BAHAYA = None

AKTIFKAN_ATURAN = False
SUMBER_AMBANG = ""  # contoh: kalibrasi lokasi/tanggal atau rujukan pembimbing
SATUAN_DAN_REFERENSI_TERKONFIRMASI = False

aturan = {"enabled": AKTIFKAN_ATURAN, "source": SUMBER_AMBANG,
          "units_and_reference_confirmed": SATUAN_DAN_REFERENSI_TERKONFIRMASI,
          "rules": []}
for sensor, direction, rawan, bahaya in [
    ("SM", "low", SM_RAWAN, SM_BAHAYA),
    ("ST", "high", ST_RAWAN, ST_BAHAYA),
    ("WL", "low", WL_RAWAN, WL_BAHAYA),
]:
    if rawan is not None or bahaya is not None:
        aturan["rules"].append({"sensor": sensor, "direction": direction,
                                "rawan": rawan, "bahaya": bahaya})
validate_rules(aturan)
print("Aturan aktif:", aturan["enabled"], "| Sensor:", [r["sensor"] for r in aturan["rules"]])
# Opsional: simpan konfigurasi ini agar juga dipakai oleh python ews_gambut.py:
# Path("aturan_risiko.json").write_text(json.dumps(aturan, indent=2), encoding="utf-8")''')
md("""## 3. Baca dan periksa data MySQL
Nama sensor dipetakan ke alias yang Anda minta. Tanggal di luar rentang baca
dicatat. Tanggal masa depan tidak dipakai untuk melatih model.
""")
code('''raw, coverage = load_mysql(start=HISTORY_START, as_of=AS_OF)
display(coverage)
print("Baris dalam rentang:", len(raw), "| Jumlah perangkat:", raw.device_id.nunique())
display(raw[["device_id", "date_time"] + FEATURES].head())''')
md("""## 4. Jalankan prediksi dan evaluasi

Input berupa 24 jam terakhir yang lengkap, dengan keluaran langsung untuk jam
ke-1 sampai ke-24. Ada enam model bersama, satu per parameter, memakai 24 jam
parameter itu sendiri. Lag dan target dibuat terpisah per perangkat, baru
digabung untuk training. Data dibagi dengan batas waktu global berurutan
menjadi train, validation, test, dengan pemisahan target 24 jam agar pelatihan
tidak membaca masa depan. Untuk SM/ST, kandidat Ridge delta, ExtraTrees delta,
musiman 24 jam dan rata-rata 6 jam dibandingkan dengan nilai terakhir.
Metode dipilih di validation per blok horizon 1-6, 7-12, 13-24 jam, dengan
minimal 2% perbaikan MAE validation. ph/SEC/WL/RR memakai nilai terakhir
berdasarkan benchmark historis. Input konstan memakai nilai terakhir.
MAE dilaporkan pada test terpisah; kebijakan rilis sudah memakai hasil benchmark
historis sehingga masih perlu diuji lagi pada data baru.

Agregasi per jam menggunakan rata-rata. **RR adalah rata-rata pembacaan alat,
bukan total hujan.** Celah tetap kosong; paket empat sensor tanah nol bersamaan
ditandai mencurigakan. Training memerlukan minimal 168 waktu asal lengkap per parameter.
Perangkat dengan input prediksi tidak cukup atau lebih lama dari batas training dilewati dan dijelaskan
di tabel audit. Tidak ada interval prediksi yang sudah dikalibrasi.
""")
code('''forecast, metrics, audit = run_all(raw, aturan, output_dir=OUTPUT_DIR, as_of=AS_OF)
display(audit[["device_id", "rows", "complete_hours", "constant_sensors", "result"]])
print("Jumlah baris prediksi:", len(forecast))
if forecast.empty:
    print("Belum ada perangkat yang memenuhi kebutuhan data; lihat alasan audit.")
else:
    display(forecast[["device_id", "forecast_origin", "data_lama", "status"]].drop_duplicates())''')
md("""## 5. Prediksi sensor 24 jam

Ganti `DEVICE_ID` untuk melihat perangkat lain yang berhasil diprediksi.
Garis penuh adalah observasi per jam; garis putus-putus adalah prediksi.
Sumbu tiap sensor memakai skalanya sendiri. Satuan ST, SEC dan RR yang belum
terverifikasi tetap ditandai. Data lama tidak menghasilkan peringatan saat ini.
""")
code('''DEVICE_ID = None  # contoh: "LPA-C2413122025-M0015"; None memilih perangkat berhasil pertama
if not forecast.empty:
    device = DEVICE_ID or forecast.device_id.iloc[0]
    predicted = forecast.loc[forecast.device_id.eq(device)].set_index("date_time")
    if predicted.empty:
        raise ValueError("Perangkat ini belum punya forecast. Pilih dari tabel audit berstatus ok.")
    history, _ = prepare_device(raw, device)
    origin = pd.Timestamp(predicted.forecast_origin.iloc[0])
    history = history.loc[origin - pd.Timedelta(hours=71):origin]
    units = {"SM": "%", "ST": "derajat (skala belum pasti)", "ph": "pH",
             "SEC": "satuan alat", "WL": "cm (makin besar = air makin tinggi)",
             "RR": "rata-rata pembacaan alat"}
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), layout="constrained")
    for ax, sensor in zip(axes.flat, FEATURES):
        ax.plot(history.index, history[sensor], label="Observasi per jam", color="#1565C0")
        joined = pd.concat([history[[sensor]].tail(1), predicted[[sensor]]])
        ax.plot(joined.index, joined[sensor], "--", label="Prediksi 24 jam", color="#B45309")
        ax.axvline(origin, color="#555555", linestyle=":", linewidth=1)
        ax.set_title(sensor)
        ax.set_ylabel(units[sensor])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\\n%H:%M"))
        ax.legend(fontsize=9)
    label = "HISTORIS / DATA LAMA" if predicted.data_lama.iloc[0] else "Prediksi eksperimen"
    fig.suptitle(f"{device} | {label} | asal prediksi {origin:%d %b %Y %H:%M}", fontsize=14)
    fig.savefig(Path(OUTPUT_DIR) / "grafik_prediksi_24_jam.png", dpi=130)
    plt.show()
    display(predicted[FEATURES + ["status", "alasan", "catatan_evaluasi"]].round(3))''')
md("""## 6. Bandingkan dengan nilai terakhir

MAE makin kecil makin baik; satuannya mengikuti sensor. Nilai pada tabel adalah
MAE gabungan perangkat untuk seluruh horizon 1–24 jam, bukan persentase akurasi.
Grafik membandingkan tiap horizon. Pemilihan di validation dapat lebih buruk
pada test, sehingga model belum otomatis layak untuk EWS operasional.
Error nol pada sensor konstan juga perlu pemeriksaan alat. Titik awal test
saling tumpang tindih; jangan menganggapnya sebagai eksperimen independen.
""")
code('''if not metrics.empty:
    summary = metrics.groupby(["sensor", "metode"])[["mae", "mae_baseline"]].mean().reset_index()
    summary["lebih_buruk_dari_baseline"] = summary.mae > summary.mae_baseline
    display(summary.round(4))
    chosen_metrics = metrics  # evaluasi gabungan seluruh perangkat per parameter
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    for ax, sensor in zip(axes.flat, FEATURES):
        part = chosen_metrics.loc[chosen_metrics.sensor.eq(sensor)]
        ax.plot(part.horizon_jam, part.mae, color="#1565C0", label="Metode terpilih")
        ax.plot(part.horizon_jam, part.mae_baseline, "--", color="#B45309", label="Nilai terakhir")
        ax.set_title(sensor)
        ax.set_xlabel("Horizon (jam)")
        ax.set_ylabel("MAE dalam satuan sensor")
        ax.legend(fontsize=9)
    fig.suptitle("Evaluasi test per parameter | seluruh perangkat", fontsize=14)
    fig.savefig(Path(OUTPUT_DIR) / "grafik_evaluasi.png", dpi=130)
    plt.show()''')
md("""## 7. Batas penggunaan dan hasil tersimpan

- `hasil_ews/prediksi_24_jam.csv`: prediksi tiap jam untuk enam sensor dan status.
- `hasil_ews/evaluasi_mae.csv`: MAE per sensor/horizon beserta baseline.
- `hasil_ews/audit_perangkat.csv`: pemeriksaan kelengkapan dan alasan gagal.
- `hasil_ews/model_parameter_{sensor}.joblib`: enam model parameter bersama dan metadata.
  Model lama per device tidak dipakai API; jalankan ulang training untuk membuat model baru.

Klasifikasi aturan dapat diaktifkan setelah variabel ambang diisi. Kolom
`data_lama` menghalangi status peringatan saat ini jika telemetri sudah lama;
hasil historis tetap bisa dipelajari. MAE sensor bukan akurasi prediksi kebakaran.
Untuk mengevaluasi Normal/Rawan/Bahaya, kumpulkan label lapangan independen dengan
`device_id`, `date_time`, `status`, dan `sumber_label`. Baca `README.md` untuk
kebijakan data dan cara menjalankan skrip.

Referensi metode: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
dan [evaluasi deret waktu](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.validate(notebook)
nbf.write(notebook, path)
print("Notebook disusun:", path)
