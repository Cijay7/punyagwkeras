---
title: EWS Gambut Keras REST
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# EWS Gambut: REST API + enam model Keras terbaik

Folder ini mandiri: tidak mengimpor file proyek dari direktori lain, tidak
memerlukan MySQL di Space, dan tidak melakukan training pada request.
Model SM, ST, ph, SEC, WL dan RR sudah disertakan bersama metadata/checksum.

Alur utama sesuai konfigurasi sumber **HTTPS**:

```text
Client -> REST API Space -> REST sumber HTTPS -> histori device
                        -> enam champion Keras -> JSON 24 jam
```

Data asli tetap berada pada backend Anda. Space hanya mengambil rentang input
terakhir untuk setiap permintaan. API sumber harus dapat diakses dari internet
melalui HTTPS; `localhost` komputer Anda bukan alamat database/source di Space.

## 1. Siapkan REST API sumber

Jika backend Anda sudah mempunyai endpoint histori, cukup sesuaikan kontrak
di bawah. Jika belum, bridge read-only `examples/source_bridge.py` sudah tersedia.
Bridge dijalankan pada komputer/server yang dapat mengakses MySQL, **bukan di Space**.

PowerShell dari folder paket:

```powershell
python -m pip install -r examples/source_bridge_requirements.txt
$env:SOURCE_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:SOURCE_DATABASE = 'tugasakhir_sintetis_10menit'
# MYSQL_URL mengikuti kredensial lokal Anda bila bukan koneksi default proyek.
python -m uvicorn source_bridge:app --app-dir examples --host 127.0.0.1 --port 8092
```

Simpan nilai `SOURCE_API_TOKEN`; nilai yang sama harus dipasang di Space.
Bridge hanya menjalankan SELECT. Endpoint `/health` tidak memerlukan token;
endpoint data memakai `Authorization: Bearer TOKEN_SUMBER`.

Untuk pengujian dari komputer lokal, buka terminal kedua dan gunakan
[Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/):

```powershell
cloudflared tunnel --url http://localhost:8092
```

Install `cloudflared` mengikuti dokumentasi resminya terlebih dahulu. Perintah
itu menampilkan URL HTTPS sementara, misalnya `https://acak.trycloudflare.com`.
Alamat sumber untuk Space menjadi `https://acak.trycloudflare.com/api/v1`.
Quick Tunnel berhenti ketika proses/komputer berhenti dan URL dapat berubah;
untuk layanan tetap gunakan REST API HTTPS atau named tunnel dengan domain tetap.
Jangan menjalankan tunnel ke port MySQL 3306: yang diekspos adalah bridge REST.

### Kontrak sumber

Space memanggil dua endpoint berikut dengan Bearer token sumber:

```text
GET {SOURCE_API_BASE_URL}/devices
GET {SOURCE_API_BASE_URL}/devices/{device_id}/history?hours=25
```

Contoh respons daftar device:

```json
{"devices":[{"device_id":"LPA-C2613122025-M0014","latest_time":"2026-05-13T06:30:00+07:00"}]}
```

Respons histori harus berbentuk seperti `examples/readings.json`:

```json
{
  "device_id": "LPA-C2613122025-M0014",
  "time_is_synthetic": true,
  "readings": [
    {"date_time":"2026-05-12T07:00:00+07:00","SM":100,"ST":28,"ph":5,"SEC":0,"WL":70,"RR":0}
  ]
}
```

Contoh singkat di atas hanya menunjukkan bentuk satu baris. Untuk prediksi,
kirim pembacaan yang mencakup **24 jam penuh terbaru**, maksimal 2000 baris.
24 pembacaan berjarak 10 menit hanya mencakup sekitar empat jam dan tidak cukup.
Semua timestamp wajib memiliki offset, misalnya `+07:00` atau `Z`. Enam field
sensor wajib hadir; nilai hilang boleh `null`, tetapi window yang tidak lengkap
tidak menghasilkan prediksi. Server mengurutkan waktu dan memilih duplikat
terakhir dalam input. Nilai future/reset clock yang tidak wajar ditolak.

## 2. Buat Hugging Face Space

Status akun Cijay67 pada 16 November 2026: pembuatan Docker Space dengan
CPU Basic ditolak oleh Hugging Face dengan HTTP 402. Pesan server menyatakan
hosting Docker/Gradio pada cpu-basic memerlukan [PRO](https://huggingface.co/pro).
Tidak ada pembelian langganan yang dilakukan. Repository model
[`Cijay67/Nyawit`](https://huggingface.co/Cijay67/Nyawit) dapat menyimpan paket ini,
tetapi tidak menjalankan REST API. Untuk menjalankan API, diperlukan Docker Space
yang diizinkan oleh akun atau host Docker lain, serta konfigurasi sumber HTTPS.

1. Masuk ke Hugging Face dan buka [Create a new Space](https://huggingface.co/new-space).
2. Pilih nama, misalnya `ews-gambut-rest`.
3. Pilih **Docker**, template **Blank**, hardware **CPU Basic**.
4. Pilih Private untuk membatasi akses, atau Public bila memang ingin API/repo
   dapat diakses publik. Pada Private Space, client programatis juga memerlukan
   token Hugging Face yang memiliki akses.

Dockerfile menjalankan Uvicorn pada `0.0.0.0:7860`, sesuai `app_port` di atas,
dengan user UID 1000 dan satu worker CPU.

## 3. Upload isi folder ke root repository Space

`Dockerfile`, `README.md` dan `app.py` harus berada langsung di root repo Space,
bukan di dalam subfolder `huggingface_ews_gambut/` lagi.

Bisa melalui **Files -> Add file -> Upload files**, dengan mempertahankan
struktur `models/`, `examples/`, dan file lainnya. Untuk upload folder secara
konsisten, gunakan CLI `hf` berikut dari direktori induk paket:

```powershell
python -m venv .venv-hf-cli
.\.venv-hf-cli\Scripts\python.exe -m pip install --upgrade huggingface_hub
.\.venv-hf-cli\Scripts\hf.exe auth login
.\.venv-hf-cli\Scripts\hf.exe repos create USERNAME/ews-gambut-rest --type space --sdk docker --private
.\.venv-hf-cli\Scripts\hf.exe upload USERNAME/ews-gambut-rest .\huggingface_ews_gambut . --type space --exclude "*.pyc" --exclude "**/__pycache__/**" --exclude "verification/**"
```

Ganti `USERNAME`. Jika Space sudah dibuat melalui web, lewati `repos create`.
`--private` adalah pilihan pada contoh CLI; untuk Public pilih visibilitas yang
sesuai saat membuat Space. Tidak ada login/token pribadi yang disimpan dalam paket.

## 4. Isi Secrets dan Variables

Di **Space -> Settings -> Variables and secrets**, tambahkan:

| Nama | Jenis | Nilai |
|---|---|---|
| `SOURCE_API_BASE_URL` | Secret | URL HTTPS sumber **termasuk `/api/v1`** |
| `SOURCE_API_TOKEN` | Secret | Token yang sama dengan bridge/backend sumber |
| `API_KEY` | Secret | Kunci pilihan Anda untuk client yang memanggil Space |
| `DATA_SOURCE` | Variable | `https` |
| `TIMEZONE` | Variable | `Asia/Jakarta`, sama dengan training |
| `MAX_AGE_MINUTES` | Variable | `120` |
| `CORS_ORIGINS` | Variable, opsional | Origin frontend browser, dipisahkan koma |

`API_KEY` mengamankan client -> Space lewat `X-API-Key`; `SOURCE_API_TOKEN`
mengamankan Space -> sumber lewat Bearer. Keduanya berbeda dari token login HF.
Tanpa `API_KEY`, API Space tidak meminta kunci tambahan. `.env.example` hanya
contoh dan tidak dimuat otomatis; jangan upload `.env` yang berisi token nyata.

## 5. Tunggu build dan coba REST API

Setelah status **Running**, buka URL aplikasi Space yang ditampilkan Hugging Face.
Gunakan URL aplikasi `*.hf.space`, bukan URL halaman repo `huggingface.co/spaces/...`.

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Model termuat dan status konfigurasi sumber |
| GET | `/docs` | Swagger UI, tombol Authorize untuk X-API-Key |
| GET | `/api/v1/models` | Sumber model pemenang setiap parameter |
| GET | `/api/v1/devices` | Daftar device dari REST sumber |
| GET | `/api/v1/devices/{device_id}/latest` | Pembacaan terbaru dari sumber |
| GET | `/api/v1/devices/{device_id}/classification` | Klasifikasi pembacaan terbaru |
| GET | `/api/v1/devices/{device_id}/predictions` | Histori dari HTTPS -> prediksi 24 jam |
| GET | `/api/v1/devices/{device_id}/results` | Sensor terakhir + prediksi |
| POST | `/api/v1/devices/{device_id}/predictions` | Alternatif: kirim histori JSON langsung |

`/health` tidak memerlukan X-API-Key. `source_configured=true` berarti URL sudah
diisi, belum membuktikan sumber dapat dijangkau. Coba `/api/v1/devices` untuk
memeriksa koneksi HTTPS dan token.

Contoh PowerShell:

```powershell
$space = 'https://URL-APLIKASI-ANDA.hf.space'
$headers = @{'X-API-Key'='API_KEY_ANDA'}
$device = 'LPA-C2613122025-M0014'
Invoke-RestMethod "$space/api/v1/devices" -Headers $headers
Invoke-RestMethod "$space/api/v1/devices/$device/predictions?mode=demo" -Headers $headers |
    ConvertTo-Json -Depth 12
```

Untuk Private Space tambahkan `Authorization: Bearer HF_READ_TOKEN` pada client,
selain X-API-Key jika dikonfigurasi. Token sumber tetap hanya berada di server.

POST alternatif tidak memakai sumber HTTPS dan tidak menyimpan data request:

```powershell
Invoke-RestMethod "$space/api/v1/devices/$device/predictions" -Method Post -Headers $headers -ContentType 'application/json' -Body (Get-Content examples/readings.json -Raw) |
    ConvertTo-Json -Depth 12
```

`device_id` di body harus sama dengan path. Contoh JSON yang disertakan berisi
24 nilai per jam dari dataset sintetis Anda, hanya untuk menguji bentuk request.

## Interpretasi hasil dan error

Default `mode=demo`: 24 titik pada `prediction.points`, masing-masing berisi
waktu, horizon, sensor dan klasifikasi aturan. Sumber input dan sumber model
dibedakan pada `data_source` dan `model_per_parameter.*.source_device_id`.
`model_training_after_input` menandai model final yang dilatih setelah origin
input; hasil demo tersebut **bukan backtest**.

Model dilatih pada waktu sintetis 10 menit. Input diagregasi rata-rata per jam,
kemudian 24 jam input menghasilkan 24 jam prediksi. RR bukan total hujan.
Tidak ada klaim akurasi kebakaran. `mode=live` tidak menerbitkan prediksi
operasional: data lama menghasilkan `stale_data`, dan input segar tetap
`synthetic_model` karena model belum tervalidasi untuk peringatan live.

HTTP 200 dapat berisi `invalid_latest_reading`, `insufficient_history`,
`invalid_time` atau `future_reading` dengan points kosong. HTTP 401 berarti
X-API-Key salah; 422 berarti schema/input tidak sesuai. Error sumber HTTPS
memakai 404/502/503 dengan alasan eksplisit, tanpa mengembalikan token/SQL.
Paket empat sensor tanah nol bersamaan tidak diganti dengan data buatan.

GET tidak menulis database atau meng-cache histori. Jika bridge lokal, MySQL,
komputer atau tunnel berhenti, GET Space akan melaporkan sumber tidak tersedia.
POST JSON tetap dapat dipakai tanpa sumber. Free/idle hardware dapat mengalami
cold start; lihat pengaturan lifecycle Space untuk kebutuhan uptime Anda.

## Uji lokal sebelum upload

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:DATA_SOURCE='example'
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Mode `example` hanya menyediakan device contoh M0014. Untuk menguji sumber
sebenarnya, set `DATA_SOURCE=https` dan URL/token HTTPS. Docker lokal opsional:

```powershell
docker build -t ews-gambut-rest .
docker run --rm -p 7860:7860 --env-file .env ews-gambut-rest
```

`.env` harus dibuat sendiri dari contoh dengan konfigurasi yang benar.
Paket tidak membawa credential dan tidak mengandalkan file di luar folder.

## Isi dan pengujian paket

`models/champions/` berisi enam `.keras` dan enam metadata JSON. Checksum
model diverifikasi sebelum load. `models/provenance.json` mencatat versi training.
`core.py`, `runtime.py`, `source.py` dan `app.py` berisi inferensi/REST saja.
Docker tidak menginstal dependensi MySQL, scikit-learn, PyTorch atau training.

Untuk tes unit, install `requirements-dev.txt`, kemudian:

```powershell
python -m unittest discover -s tests -v
```

Catatan hasil pengujian paket ada di `VERIFICATION.md`. Build image Docker dan
deployment remote tetap perlu dikonfirmasi pada platform tujuan; panduan ini
tidak berarti paket sudah dipublikasikan ke akun Hugging Face Anda.

Referensi resmi:
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[jaringan/lifecycle Spaces](https://huggingface.co/docs/hub/spaces-overview),
[CLI hf](https://huggingface.co/docs/huggingface_hub/en/guides/cli), dan
[Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
