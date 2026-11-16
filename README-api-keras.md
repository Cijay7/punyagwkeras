# API lokal Keras dengan Uvicorn

API baru ada di `api_keras.py`; API/model/training sebelumnya tetap terpisah.
Input berasal dari device yang dipilih, sedangkan SM, ST, ph, SEC, WL, RR
masing-masing memakai model pemenang dari `hasil_ews_keras/model_terbaik.json`.
Tidak ada training atau penulisan database ketika endpoint dipanggil.

## Menjalankan

Jalankan dari PowerShell di direktori proyek:

```powershell
python -m uvicorn api_keras:app --host 127.0.0.1 --port 8091
```

Atau:

```powershell
.\start_api_keras.ps1
```

Jika port sudah digunakan, hentikan server sebelumnya dengan Ctrl+C pada
terminalnya, atau gunakan `.\start_api_keras.ps1 -Port 8092`.
Tidak perlu menjalankan notebook atau training ulang untuk setiap request.
Dependensi tersedia melalui `requirements-keras.txt`, yang juga menyertakan
FastAPI/Uvicorn dari requirements proyek.

Buka dokumentasi interaktif [Swagger](http://127.0.0.1:8091/docs), pilih
endpoint, klik **Try it out**, isi `device_id`, lalu **Execute**.
Schema tersedia di [OpenAPI](http://127.0.0.1:8091/openapi.json).

## Endpoint

| Method | Endpoint | Isi |
|---|---|---|
| GET | `/health` | Koneksi database dan identitas layanan |
| GET | `/api/v1/models` | Device asal model terbaik tiap parameter, waktu training dan MAE |
| GET | `/api/v1/devices` | Daftar device beserta waktu pembacaan terbaru |
| GET | `/api/v1/devices/{device_id}/latest` | Pembacaan sensor terakhir dan klasifikasi |
| GET | `/api/v1/devices/{device_id}/classification` | Klasifikasi pembacaan terakhir |
| GET | `/api/v1/devices/{device_id}/predictions` | Prediksi 24 jam dari enam champion Keras |
| GET | `/api/v1/devices/{device_id}/results` | Pembacaan terakhir, klasifikasi dan prediksi dalam satu respons |

Contoh:

```powershell
$device = 'LPA-C2413122025-M0015'
Invoke-RestMethod "http://127.0.0.1:8091/api/v1/devices/$device/predictions?mode=demo" |
    ConvertTo-Json -Depth 12

Invoke-RestMethod "http://127.0.0.1:8091/api/v1/devices/$device/results" |
    ConvertTo-Json -Depth 12
```

Setiap titik di `prediction.points` berisi `date_time`, `horizon_jam`, enam
nilai pada `sensors`, serta `classification`. Timestamp menggunakan offset WIB
`+07:00`. Origin adalah jam lengkap terakhir, bukan waktu request. Sebanyak 24
jam input terakhir diagregasi sama seperti training. RR adalah rata-rata
pembacaan alat, bukan total hujan.

## Mode demo dan live

Default **`mode=demo`** disediakan untuk mencoba database waktu sintetis yang
sudah lama. Mode ini boleh menghasilkan keluaran dari input historis memakai
model final, dengan penanda yang selalu dikembalikan:

- `is_demo=true`, `is_backtest=false`, `is_operational=false`.
- `time_is_synthetic=true`: waktu dalam database dibuat ulang 10 menit per baris.
- `input_data_lama`: apakah input sudah lebih lama dari batas kesegaran.
- `model_training_after_input`: apakah setidaknya satu model dilatih setelah
  origin input. Rinciannya ada pada tiap `model_per_parameter`.
- `model_per_parameter.{sensor}.source_device_id`: device asal model pemenang,
  yang dapat berbeda dari device input.

`demo_ready` berarti JSON prediksi berhasil dibuat untuk demonstrasi, **bukan
prediksi kondisi sekarang atau bukti akurasi backtest**. Beberapa model final
dilatih melewati tanggal input device lain, sehingga demo tersebut tidak boleh
dinilai sebagai evaluasi pada data yang belum pernah dilihat. Klasifikasi titik
dengan data lama tetap `Data lama`. Ambang tidak aktif menghasilkan `Belum
dikonfigurasi` bila data tidak lama.

Untuk memeriksa kesiapan live:

```powershell
Invoke-RestMethod "http://127.0.0.1:8091/api/v1/devices/$device/predictions?mode=live"
```

`mode=live` menolak data lama, input sintetis, histori tidak lengkap dan model
dari masa depan. Pada database sekarang hasilnya `stale_data` tanpa points;
meskipun dibuat seolah segar, data sintetis tetap ditolak (`synthetic_data`).
API ini memang ditujukan ke dataset sintetis; metadata dataset diperiksa pada
setiap pembacaan. Penggunaan operasional dengan data asli memerlukan model dan
validasi yang sesuai, bukan sekadar mengganti parameter mode.

## Status dan error

| Status | Arti |
|---|---|
| `demo_ready` | Ada 24 titik demo, dengan batas interpretasi di atas |
| `stale_data` | Mode live menunggu data segar |
| `synthetic_data` | Data sintetis tidak boleh menjadi peringatan live |
| `invalid_latest_reading` | SM/ST/ph/SEC terakhir nol bersamaan |
| `insufficient_history` | Ada gap atau sensor kosong pada 24 jam input terbaru |
| `model_not_found` | Manifest atau enam pasangan model/metadata belum lengkap |
| `model_invalid` | Checkpoint, metadata, pemilihan champion atau keluaran tidak valid |
| `model_from_future` | Pada mode live, training melewati origin input |

Kondisi data/model belum siap memakai HTTP 200 dengan status dan alasan.
HTTP 404 berarti device tidak ditemukan, 422 berarti input/mode tidak valid,
dan 503 berarti database/aturan/metadata dataset tidak tersedia atau tidak
sesuai. Sensor kosong/tidak finite pada pembacaan terakhir menjadi JSON `null`.
Histori yang melebihi batas baris ditolak, bukan dipotong diam-diam.

Pada pengujian dataset yang tersedia, 8 device menghasilkan 24 titik demo.
M0013, M0016, P0007 dan P0009 ditolak karena paket sensor tanah terakhir nol
bersamaan. API tidak mengarang data untuk melewati kondisi ini.

## Data terbaru, cache dan aturan

Setiap request membaca MySQL kembali. Endpoint prediksi hanya mengambil
rentang 24 jam input per device, bukan semua histori training. Cache menyimpan
model dan hasil komputasi; perubahan manifest/checkpoint/metadata model atau
isi window input membatalkan cache yang sesuai. Respons HTTP selalu `no-store`.
Inferensi model diserialkan dengan lock untuk penggunaan CPU lokal.

`aturan_risiko.json` dibaca ulang pada request. Saat ini aturan belum aktif;
API tidak membuat ambang baru. Data/model sintetis dan MAE sensor juga tidak
membuktikan akurasi prediksi kebakaran.

## Konfigurasi

Default: database `tugasakhir_sintetis_10menit`, timezone `Asia/Jakarta`, model
`hasil_ews_keras/`, usia maksimum 120 menit, dan batas histori 50000 baris.
File `.env` tidak dimuat otomatis.

| Environment | Fungsi |
|---|---|
| `EWS_KERAS_MYSQL_URL` | URL database penuh khusus API Keras |
| `MYSQL_URL` | Kredensial/host jika URL khusus tidak diisi; nama database tetap dari `EWS_KERAS_DATABASE` |
| `EWS_KERAS_DATABASE` | Nama database, default salinan sintetis |
| `EWS_KERAS_MODEL_DIR` | Folder hasil training berisi manifest dan subfolder champions |
| `EWS_KERAS_RULES_PATH` | Lokasi aturan risiko |
| `EWS_KERAS_TIMEZONE` | Harus cocok dengan metadata dataset |
| `EWS_KERAS_MAX_AGE_MINUTES` | Batas kesegaran untuk mode live |
| `EWS_KERAS_MAX_HISTORY_ROWS` | Batas kueri histori sebelum inferensi |

## Tes dan bukti integrasi

```powershell
python -m unittest test_api_keras.py
```

Pengujian HTTP dengan database dan enam checkpoint asli disimpan di
`verifikasi_api_keras/verifikasi_http.json`. Contoh respons lengkap untuk
frontend ada di `verifikasi_api_keras/contoh_results.json`.

Referensi: [FastAPI dan dokumentasi interaktif](https://fastapi.tiangolo.com/tutorial/first-steps/),
[pengaturan Uvicorn](https://uvicorn.dev/settings/).
