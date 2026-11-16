# GRU per device dan per parameter

`ews_gambut_pytorch_device.py` melatih model terpisah untuk setiap pasangan
device-parameter yang memiliki histori cukup. Contoh: SM dari M0015 dilatih
terpisah dari SM milik M0019. Hasil akhirnya adalah **satu pemenang untuk setiap
parameter** SM, ST, ph, SEC, WL, dan RR.

## Menjalankan dan melanjutkan setelah terhenti

Dari direktori proyek, memakai environment PyTorch yang sudah disiapkan:

```powershell
.\.venv-pytorch\Scripts\python.exe ews_gambut_pytorch_device.py --snapshot uji_model/hourly_snapshot.pkl
```

Jika komputer atau proses terhenti, tambahkan `--resume` dengan sumber dan
konfigurasi yang sama:

```powershell
.\.venv-pytorch\Scripts\python.exe ews_gambut_pytorch_device.py --snapshot uji_model/hourly_snapshot.pkl --resume
```

Checkpoint lengkap digunakan kembali setelah data, konfigurasi, skala dan
metadata diperiksa. Tahap yang belum selesai atau checkpoint yang rusak dilatih
ulang. Resume bekerja pada batas tahap validation, evaluasi dan final; tidak
melanjutkan dari tengah epoch. Penulisan checkpoint memakai file sementara
yang kemudian diganti secara atomik. `run.json` mencatat status dan hash sumber.

Untuk eksperimen berbeda, gunakan folder baru, misalnya
`--output hasil_device_eksperimen_2 --epochs 100`. Resume menolak konfigurasi
atau snapshot berbeda dari catatan run sebelumnya.

Tanpa `--snapshot`, skrip membaca MySQL secara read-only dan menyimpan snapshot
hourly di folder output. Resume tanpa `--snapshot` menggunakan snapshot tersebut,
bukan pembacaan MySQL baru. Model dan API versi sebelumnya tetap terpisah.

## Syarat dan pemilihan pemenang

- Arsitektur GRU, 24 jam input dan 24 jam output, mengikuti versi PyTorch awal.
- Setiap kandidat hanya menerima data dari **satu device dan satu parameter**,
  termasuk saat normalisasi dan fit ulang model evaluasi/final.
- Semua kandidat suatu parameter memakai batas waktu global yang sama, dengan
  gap target 24 jam sebelum validation dan test.
- Default minimal 168 jendela training dan 24 jendela validation pada device
  asal. Device tanpa histori cukup dicatat dalam audit.
- Sensor dengan target perubahan training konstan tetap mendapatkan checkpoint,
  tetapi hanya dilatih satu epoch dan **tidak boleh menjadi pemenang**.
- Device pembanding harus punya minimal 24 origin train dan validation serta
  target perubahan training yang bervariasi. Kriteria ini tidak memakai test.
- Epoch dipilih pada validation device asal. Setelah seluruh kandidat selesai,
  tiap kandidat diprediksi pada **baris validation pembanding yang sama** lintas
  device. Skor ranking adalah rata-rata MAE per device, dengan bobot sama untuk
  setiap device pembanding (macro MAE).
- Pemenang dikunci berdasarkan macro MAE validation terendah. Baru setelah itu
  kandidat di-fit ulang pada histori device asal sebelum test, untuk mengukur
  MAE test. Test tidak memilih pemenang.
- Model final dilatih pada semua jendela lengkap **device asalnya sendiri**,
  dengan jumlah epoch yang telah dipilih. Model final tidak menghitung metrik test.

Pemenang antar-device bisa berbeda untuk SM dan ST, misalnya. Pemenang berarti
terbaik dari kandidat yang memenuhi syarat, bukan jaminan mengalahkan baseline
atau model gabungan. Jika tidak ada kandidat yang layak, parameter ditandai
belum tersedia dan tidak dibuat pemenang palsu. Satuan atau kalibrasi yang
berbeda antar-device tetap perlu dikonfirmasi sebelum menerapkan model lintas
device di lapangan.

## Hasil

Default folder: `hasil_ews_pytorch_device/`.

| Hasil | Isi |
|---|---|
| `model_terbaik.csv` / `model_terbaik.json` | Satu pemenang per parameter, device asal, skor validation/test dan lokasi model |
| `ranking_model.csv` | Seluruh kandidat, kelayakan ranking, MAE pada device asal dan MAE pembanding bersama |
| `audit_device_parameter.csv` | Setiap pasangan device-parameter dan alasan dilatih/dilewati |
| `evaluasi_horizon.csv` | MAE test per horizon untuk tiap kandidat pada origin pembanding bersama; agregasi berbobot origin |
| `hasil_pemilihan.md` | Ringkasan hasil beserta batas kesimpulan |
| `ranking_validation.png` | Perbandingan kandidat berdasarkan skor validation macro |
| `prediksi_champions.csv` | Prediksi 24 jam per device menggunakan keenam model pemenang |
| `audit_prediksi.csv` | Device yang dapat diprediksi atau alasan tidak tersedia |
| `run.json` | Konfigurasi, hash snapshot dan status proses |

Checkpoint tiap pasangan tersimpan di:

```text
devices/{device_id}/validation_parameter_{sensor}.pt
devices/{device_id}/evaluasi_parameter_{sensor}.pt
devices/{device_id}/model_parameter_{sensor}.pt
```

Model pemenang disalin ke `champions/` dengan nama yang sama. Checkpoint
validation untuk memeriksa seleksi, checkpoint evaluasi untuk mereproduksi
test, dan checkpoint final untuk inferensi setelah training. Metadata mencatat
`device_id`, `training_devices`, batas waktu dan fingerprint data fit.

Rincian MAE masing-masing device pembanding tersimpan di
`devices/{device_id}/evaluasi_{sensor}.json`. Perhatikan bahwa MAE test macro
pada tabel utama memberi bobot sama per device, sedangkan `evaluasi_horizon.csv`
memberi bobot sama per origin. Kedua agregasi ini memang berbeda.

Untuk menggunakan keenam pemenang dari Python:

```python
from ews_gambut_pytorch_device import forecast_champions

# hourly: DataFrame satu device dari prepare_device, dengan indeks per jam.
# aturan: konfigurasi aturan_risiko.json yang sudah dibaca.
# result = forecast_champions(hourly, "hasil_ews_pytorch_device", aturan, as_of=...)
```

Prediksi tetap memeriksa 24 jam input terbaru dan menandai data lama. Model yang
dilatih melewati waktu input ditolak. API yang ada masih menggunakan model
klasik; pemilihan pemenang ini belum mengganti model API.

## Pemeriksaan

```powershell
.\.venv-pytorch\Scripts\python.exe -m unittest test_ews_gambut_pytorch_device.py
```

Pengujian mencakup isolasi data training antar-device, seleksi tanpa memakai
test, pembobotan macro, pengecualian sensor konstan, pemuatan checkpoint, dan
resume ketika salah satu checkpoint final belum tersimpan.

Histori pendek, gap dan sensor konstan membatasi kesimpulan. Pemilihan epoch
dan kandidat pada validation yang sama bisa terlalu menyesuaikan validation;
konfirmasi pada data masa depan masih diperlukan. MAE sensor bukan akurasi
prediksi kebakaran.
