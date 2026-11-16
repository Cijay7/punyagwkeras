# EWS gambut versi PyTorch

Implementasi terpisah di `ews_gambut_pytorch.py` untuk membandingkan GRU dengan
model klasik. Enam model dilatih per parameter (SM, ST, ph, SEC, WL, RR) memakai
jendela seluruh device. Urutan setiap device tetap terpisah saat membuat lag
dan target. Prediksi tetap tersedia per device.

## Instalasi dan menjalankan

Environment pengujian `.venv-pytorch` sudah disiapkan dengan PyTorch CPU.
Untuk menjalankan pada snapshot yang sama dengan benchmark sebelumnya:

```powershell
.\.venv-pytorch\Scripts\python.exe ews_gambut_pytorch.py --snapshot uji_model/hourly_snapshot.pkl
```

Untuk membaca data terbaru dari MySQL, gunakan konfigurasi `MYSQL_URL` yang sama:

```powershell
.\.venv-pytorch\Scripts\python.exe ews_gambut_pytorch.py
```

Database hanya dibaca. Default hasil ada di `hasil_ews_pytorch/`; model dan
hasil klasik di `hasil_ews/` tetap tersedia. API sekarang tetap memakai model
klasik; skrip ini merupakan eksperimen pembanding yang dijalankan offline.

Jika menyiapkan environment baru:

```powershell
python -m venv .venv-pytorch
.\.venv-pytorch\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-pytorch\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
```

`requirements-pytorch.txt` juga tersedia untuk instalasi memakai indeks pip
default. Versi ini berjalan pada CPU, jadi tidak memerlukan GPU/CUDA.

## Cara membandingkan hasil

`hasil_ews_pytorch/perbandingan_model.csv` berisi rata-rata MAE horizon 1-24 jam:

| Kolom | Arti |
|---|---|
| `mae_gru` | GRU murni, tanpa mengganti hasilnya dengan baseline |
| `mae_selected` | GRU atau nilai terakhir, dipilih memakai validation |
| `mae_baseline` | Mempertahankan nilai sensor terakhir untuk seluruh horizon |
| `mae_classical` | Model klasik saat ini, dilatih dan dievaluasi ulang pada data serta split yang sama |
| `penurunan_gru_vs_baseline_pct` | Positif berarti MAE GRU lebih kecil; kosong jika MAE baseline nol |

MAE lebih kecil berarti lebih baik, dengan satuan parameter aslinya.
Persentase penurunan MAE bukan persentase akurasi. Jangan membandingkan angka
antarparameter yang berbeda satuan. Lihat juga grafik `perbandingan_mae.png`.
Kolom GRU murni tetap dilaporkan walaupun kalah dari baseline.

Untuk memeriksa hasil lebih rinci:

- `evaluasi_mae.csv`: MAE per parameter dan horizon, termasuk input bervariasi.
- `evaluasi_per_device.csv`: MAE per parameter dan device.
- `prediksi_test.csv`: observasi aktual dan prediksi pada setiap origin test.
- `training_{sensor}.csv`: loss training dan MAE validation setiap epoch.
- `metadata.json`: batas split, jumlah jendela, epoch terpilih dan metode per horizon.
- `run.json`: sumber snapshot/hash, konfigurasi dan versi PyTorch.
- `prediksi_24_jam.csv`: kolom SM/ST/ph/SEC/WL/RR adalah GRU murni; kolom
  `*_selected` adalah hasil pilihan GRU/baseline. Status klasifikasi mengacu ke
  GRU murni dan mengikuti konfigurasi aturan yang ada. Data lama tetap ditandai.
- `audit_perangkat.csv`: alasan device dapat/tidak dapat diprediksi.

## Model dan pemisahan waktu

Arsitektur: GRU satu layer, hidden size 16, lalu linear dengan 24 output.
Input berisi perubahan sensor relatif nilai terakhir, sinus jam dan cosinus
jam. Urutan input dimulai dari jam terlama sampai jam terbaru. Output adalah
perubahan dari nilai terakhir untuk tiap horizon, kemudian dikembalikan ke
satuan sensor dan dibatasi dengan aturan fisik yang sama dengan model klasik.

Training memakai AdamW, L1 loss, gradient clipping, seed 42 dan satu thread CPU.
Default maksimal 60 epoch dengan patience 8. Skala normalisasi dihitung hanya
dari target perubahan pada bagian training. Input konstan tetap diproses GRU;
tidak ada penggantian diam-diam pada kolom prediksi GRU murni.

Split waktu global per parameter mengikuti model klasik: sekitar 60%/20%/20%,
dengan target training berakhir sebelum validation dan target validation
berakhir sebelum test. Minimal 168 origin unik lengkap. Early stopping hanya
memakai validation. Epoch terpilih kemudian dipakai untuk fit ulang sebelum
test, tanpa melihat label test. Skala dihitung ulang dari bagian fit tersebut.

Pilihan GRU/baseline memakai minimal 2% perbaikan MAE validation pada blok
horizon 1-6, 7-12 dan 13-24. Pemilihan ini tidak menjamin test membaik.
Model final dilatih dengan semua jendela lengkap untuk jumlah epoch yang sudah
dipilih. Model final tidak digunakan menghitung ulang metrik test.

`model_parameter_{sensor}.pt` adalah checkpoint final untuk forecast baru.
`evaluasi_parameter_{sensor}.pt` adalah checkpoint sebelum test untuk
mereproduksi evaluasi historis. Checkpoint menyimpan `state_dict`, skala,
arsitektur dan metadata. Muat dengan helper berikut:

```python
from ews_gambut_pytorch import load_checkpoint, forecast_device

bundles = {
    sensor: load_checkpoint(f"hasil_ews_pytorch/model_parameter_{sensor}.pt")
    for sensor in ["SM", "ST", "ph", "SEC", "WL", "RR"]
}
# hourly adalah data satu device hasil prepare_device dengan indeks per jam.
# forecast_device(hourly, bundles, aturan, as_of=...) menghasilkan DataFrame.
```

Pengaturan lain tersedia di `--help`, misalnya `--epochs 100 --patience 12`,
`--hidden-size 32`, `--seed 123`, `--output eksperimen_gru_2` dan
`--skip-classical`. Mengubah konfigurasi setelah melihat test merupakan
eksplorasi historis; gunakan periode baru untuk kesimpulan akurasi berikutnya.
Sensor konstan, histori pendek, gap dan origin yang tumpang tindih tetap
membatasi kesimpulan. MAE sensor tidak mengukur akurasi kebakaran.

## Pemeriksaan

```powershell
.\.venv-pytorch\Scripts\python.exe -m unittest test_ews_gambut_pytorch.py
```

Referensi: [GRU PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
dan [penyimpanan state_dict](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## Model terpisah per device dengan satu pemenang per parameter

Versi ini tersedia di `ews_gambut_pytorch_device.py`.
Lihat [panduan model per device](README-pytorch-device.md) untuk training, ranking,
checkpoint pemenang dan melanjutkan proses dengan `--resume` setelah terhenti.
