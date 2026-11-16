# Training Keras untuk database sintetis

File baru: `ews_gambut_keras.py`. Skrip ini membaca database
`tugasakhir_sintetis_10menit` secara read-only. Kode dan model versi sebelumnya
tetap terpisah. TensorFlow menjadi backend Keras dan pemrosesan memakai CPU.

## Menjalankan

```powershell
python ews_gambut_keras.py
```

Jika dependensi belum terpasang:

```powershell
python -m pip install -r requirements-keras.txt
```

Kredensial mengikuti environment variable `MYSQL_URL`, tetapi nama database
ditentukan secara eksplisit oleh `--database`, dengan default salinan sintetis.
Ini mencegah penggunaan database waktu asli secara tidak sengaja.

Untuk melanjutkan setelah proses atau komputer terhenti:

```powershell
python ews_gambut_keras.py --resume
```

Resume menggunakan snapshot hourly yang sudah tersimpan. Konfigurasi, versi
Keras/TensorFlow, hash snapshot, fingerprint data fit dan pasangan file model /
metadata diperiksa sebelum checkpoint digunakan kembali. Tahap yang belum
selesai dilatih ulang; resume bukan pemulihan dari tengah epoch.

Untuk eksperimen lain:

```powershell
python ews_gambut_keras.py --output hasil_keras_eksperimen_2 --epochs 60 --patience 8
```

## Skema training

1. Periksa bahwa database ditandai sintetis, nomor urut setiap device benar,
   intervalnya sesuai metadata, serta `date_time` dan `unixtime` konsisten di WIB.
2. Agregasikan pembacaan setiap 10 menit menjadi **rata-rata per jam** per device.
   Paket SM/ST/ph/SEC nol bersamaan dikeluarkan dengan fungsi persiapan lama.
   Gap tidak diisi dari masa depan. RR tetap rata-rata pembacaan, bukan total hujan.
3. Bentuk 24 jam input untuk memprediksi langsung 24 jam berikutnya. Enam
   parameter: SM, ST, ph, SEC, WL, RR. Setiap device-parameter menerima model
   terpisah, tanpa menggabungkan data training device lain.
4. Gunakan batas waktu global sekitar 60% train, 20% validation, 20% test per
   parameter. Target train harus berakhir sebelum validation, dan target
   validation sebelum test, dengan horizon 24 jam.
5. Default minimal 168 jendela train dan 24 jendela validation per device.
   Sensor training konstan disimpan sebagai model satu epoch, tetapi tidak
   boleh menjadi kandidat pemenang.
6. Epoch dipilih pada validation device asal. Semua kandidat kemudian diberi
   skor pada **baris validation pembanding yang sama** lintas device. Skor
   adalah rata-rata MAE per device dengan bobot sama (macro MAE).
7. Pilih satu pemenang per parameter dari validation. Baru kemudian model
   evaluasi di-fit ulang memakai histori device asal sebelum test, dengan
   jumlah epoch yang sudah dipilih. Test tidak memilih kandidat atau epoch.
8. Model final di-fit ulang pada seluruh histori device asal untuk kebutuhan
   prediksi setelah training; model final tidak dipakai menghitung MAE test.

Arsitektur: GRU satu layer hidden size 16, kemudian Dense 24 output. Input
memakai perubahan relatif dari nilai terakhir serta sinus/cosinus jam. Target
adalah perubahan setiap horizon dari nilai terakhir. Skala normalisasi hanya
dihitung dari bagian fit device asal, lalu output dikembalikan ke satuan asli.
Optimizer AdamW dengan gradient clipping, loss MAE, seed 42, default maksimal
30 epoch dan patience 6. Training memakai `train_on_batch`, sehingga tidak
membuat pool thread `tf.data` pada komputer lokal.

## Hasil

Default folder: `hasil_ews_keras/`.

| File | Isi |
|---|---|
| `model_terbaik.csv` / `model_terbaik.json` | Satu pemenang tiap parameter, device asal dan skor validation/test |
| `ranking_model.csv` | Seluruh kandidat, kelayakan, MAE device asal dan macro MAE |
| `audit_data.csv` | Persiapan data per device |
| `audit_device_parameter.csv` | Alasan pasangan dilatih atau dilewati |
| `devices/{device}/validation_{sensor}.keras` | Model tahap seleksi, hanya belajar dari blok train |
| `devices/{device}/evaluasi_{sensor}.keras` | Model sebelum test untuk mereproduksi metrik test |
| `devices/{device}/model_{sensor}.keras` | Model final dari satu device-parameter |
| `champions/` | Salinan model pemenang dan metadata JSON, untuk ketiga tahap |
| `training_{sensor}.csv` di folder device | Loss training dan MAE validation tiap epoch |
| `hasil_evaluasi_{sensor}.json` di folder device | Rincian MAE masing-masing device pembanding |
| `prediksi_24_jam.csv` | Prediksi memakai enam pemenang, per device yang memiliki input cukup |
| `audit_prediksi.csv` | Alasan prediksi tersedia / tidak tersedia |
| `hasil_training.md` dan `ranking_validation.png` | Ringkasan dan grafik ranking |
| `run.json` dan `hourly_snapshot.pkl` | Versi, konfigurasi, identitas dataset dan snapshot untuk resume |

Setiap file `.keras` mempunyai metadata `.json` di sampingnya; simpan keduanya
bersama. Checkpoint menyimpan format Keras asli, tanpa kelas atau loss custom.

```python
from ews_gambut_keras import load_bundle, forecast

model, info = load_bundle("hasil_ews_keras/champions/model_SM.keras")
# hourly adalah data satu device dengan indeks per jam dari prepare_device.
# result = forecast(hourly, "hasil_ews_keras", as_of=...)
```

MAE lebih kecil berarti lebih baik. Pemenang berarti terbaik di antara kandidat
yang memenuhi syarat, bukan jaminan mengalahkan baseline. Sensor konstan tidak
membuktikan akurasi model. Prediksi menandai waktu sintetis dan data lama; model
dari masa depan terhadap waktu input ditolak. Versi ini tidak mengganti model API.

**Data waktunya sintetis.** Hasil tidak boleh ditafsirkan sebagai bukti akurasi
prediksi lapangan atau kebakaran, dan tidak langsung dibandingkan dengan angka
benchmark terdahulu yang memakai waktu asli. Pemilihan epoch dan kandidat pada
validation yang sama juga masih memerlukan konfirmasi pada data baru.

## Tes

```powershell
python -m unittest test_ews_gambut_keras.py
```

Tes mencakup isolasi device, gap waktu, urutan input, scaler training saja,
penyimpanan/pemuatan `.keras`, resume checkpoint, timezone sintetis dan seleksi
pemenang berdasarkan validation.

Referensi: [GRU Keras](https://keras.io/api/layers/recurrent_layers/gru/),
[train_on_batch](https://keras.io/api/models/model_training_apis/), dan
[format model .keras](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/).
