# EWS lahan gambut

Lanjutkan melalui `kecerdasancijay.ipynb`, jalankan dari atas ke bawah.
Alternatif terminal: `python -m pip install -r requirements.txt` lalu
`python ews_gambut.py`. Database hanya dibaca; tidak ada perubahan tabel.
Koneksi default mengikuti notebook awal (`localhost:3306/tugasakhir`).
Untuk koneksi lain, atur environment variable `MYSQL_URL`; hindari memasukkan
password ke notebook atau membagikan nilai variabel tersebut.

## REST API: data terbaru dari MySQL

Jalankan dari PowerShell di direktori proyek:

```powershell
python -m pip install -r requirements.txt
$env:DB_TIMEZONE = 'Asia/Jakarta'
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Atau jalankan `./start_api.ps1`. Dokumentasi interaktif tersedia di
`http://127.0.0.1:8000/docs` dan skema OpenAPI di `/openapi.json`.
`DB_TIMEZONE` adalah zona waktu nilai `date_time` yang disimpan di MySQL.
Gunakan `Asia/Makassar` jika database menyimpan waktu WITA. API mengembalikan
timestamp ISO 8601 dengan offset zona waktu tersebut. `.env.example` memuat
pilihan environment; file itu tidak dimuat otomatis.

| Method | Endpoint | Hasil |
|---|---|---|
| GET | `/health` | Cek layanan dan koneksi MySQL |
| GET | `/api/v1/devices` | Daftar perangkat dan waktu pembacaan terbaru |
| GET | `/api/v1/devices/{device_id}/latest` | Enam sensor terbaru dan klasifikasinya |
| GET | `/api/v1/devices/{device_id}/classification` | Klasifikasi pembacaan sensor terbaru |
| GET | `/api/v1/devices/{device_id}/predictions` | Prediksi tiap jam untuk 24 jam dan klasifikasi prediksi |
| GET | `/api/v1/devices/{device_id}/results` | Gabungan pembacaan terbaru, klasifikasi, dan prediksi |

Contoh permintaan:

```powershell
$device = 'LPA-C2413122025-M0015'
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/devices/$device/results" |
    ConvertTo-Json -Depth 10
```

**Setiap GET membaca database lagi.** Tidak perlu menjalankan notebook setiap
ada data baru. Frontend/alat pemanggil dapat memanggil endpoint ini berkala,
misalnya setiap 15 detik. API tidak melakukan polling database saat tidak ada
request. Pembacaan terbaru ditentukan oleh `date_time DESC, id DESC`, bukan
semata-mata ID terbesar; timestamp masa depan tidak dipakai.

Untuk prediksi, API mengambil data perangkat tersebut dari 24 jam sebelum
jam lengkap terakhir sampai pembacaan terbaru. API **tidak membaca ulang seluruh
histori**, tidak membaca CSV prediksi lama, tidak melatih model pada request,
dan tidak menulis ke database. Model dibaca dari enam file
`hasil_ews/model_parameter_{sensor}.joblib` (SM, ST, ph, SEC, WL, RR)
yang dihasilkan oleh notebook/skrip pelatihan. Setiap model dipakai bersama
oleh seluruh device, termasuk device baru yang memiliki 24 jam input lengkap.
Model dilatih ulang offline jika kinerjanya perlu diperbarui.

**Migrasi:** jalankan ulang `python ews_gambut.py` atau notebook dari atas
untuk membuat keenam model parameter. File model lama per device tidak dipakai
API dan tidak dikonversi otomatis. Restart API yang sedang berjalan setelah
memperbarui kode. Output notebook lama dibersihkan agar hasil training sebelumnya
tidak tertukar dengan hasil baru.
Cache hanya menyimpan komputasi/model; perubahan data input atau file model
membatalkan cache yang relevan. Kesegaran dihitung ulang pada setiap request.

`prediction.availability` menjelaskan kondisi keluaran:

- `ready`: `points` berisi 24 prediksi, masing-masing dengan `sensors` dan
  `classification`. Waktu prediksi dihitung dari `forecast_origin`, yaitu jam
  lengkap terakhir, bukan dari detik request. Hujan RR tetap rata-rata pembacaan.
- `stale_data`: data/jam input lengkap melebihi `EWS_MAX_AGE_MINUTES` (default 120).
  `points` kosong; klasifikasi pembacaan lama adalah `Data lama`.
- `insufficient_history`: 24 jam terbaru memiliki celah atau sensor kosong.
- `model_not_found`: enam file model parameter bersama belum lengkap.
- `invalid_latest_reading`: paket SM/ST/ph/SEC terbaru nol bersamaan.
- `model_invalid` / `model_from_future`: file model tidak cocok/rusak, atau waktu
  pelatihan model melewati waktu input yang akan diprediksi.

Kondisi menunggu data/model tetap HTTP 200 dengan status dan alasan eksplisit.
HTTP 404 berarti perangkat belum punya data berwaktu valid, HTTP 422 berarti
ID tidak valid, HTTP 503 berarti database/konfigurasi tidak tersedia. Keluaran
sensor tidak terhingga/kosong menjadi JSON `null`. Batas baris histori default
50.000; jika terlampaui, API mengembalikan 503 `history_limit_exceeded` dan tidak
membuat prediksi dari histori yang terpotong.

**Ambang API diedit di `aturan_risiko.json`**. Perubahan file terbaca pada request
berikutnya tanpa restart. Bila mengubah variabel di notebook, jalankan baris
penyimpanan konfigurasi yang disediakan agar perubahan juga dipakai API.
Tanpa ambang aktif, statusnya `Belum dikonfigurasi`; API tidak menebak kategori.
Nilai SM/ST/WL dan aturan yang sudah disiapkan sebelumnya tetap dipertahankan.

Untuk mempercepat pembacaan saat tabel membesar, indeks gabungan
`(device_id, date_time, id)` dapat ditambahkan oleh pengelola database.
API tidak membuat indeks atau mengubah struktur tabel secara otomatis.

Pemeriksaan API dan logika: `python -m unittest test_ews_gambut.py test_api.py`.
Referensi implementasi: [FastAPI TestClient](https://fastapi.tiangolo.com/reference/testclient/)
dan [SQLAlchemy connection pooling](https://docs.sqlalchemy.org/en/20/core/pooling.html).

## Data dan prediksi

- Pemetaan: `soil_moisture` → SM, `soil_temperature` → ST, `ph_value` → ph,
  `soil_conductivity` → SEC, `water_level` → WL, `rain_meter` → RR.
- SM: persen; ST: derajat, skala perlu dikonfirmasi; WL: cm, makin besar berarti
  air makin tinggi menurut pengguna; nol referensi belum dikonfirmasi.
  Satuan SEC dan definisi RR belum diketahui.
- Pembersihan dan pembuatan lag/target tetap terpisah per perangkat agar tidak
  menyambung urutan dua device. Training menggabungkan jendela dari seluruh
  device untuk setiap parameter. Rata-rata per jam memakai bin berlabel
  waktu akhir. Jam parsial terakhir dibuang. RR adalah rata-rata pembacaan alat,
  **bukan total hujan per jam/24 jam**. Tetapkan agregasi yang benar setelah diketahui
  apakah RR pembacaan sesaat, tambahan per interval, atau penghitung kumulatif.
- Waktu disimpan sebagaimana `date_time` di MySQL. Zona waktunya belum diketahui;
  batas `as_of` default memakai waktu lokal Asia/Makassar tanpa konversi database.
  Konfirmasikan zona waktu sebelum integrasi operasional.
- Filter awal: 1 Januari 2026 sampai waktu eksekusi; tanggal di luar rentang
  dicatat, tidak diperbaiki/dihapus di MySQL. Sesuaikan `start` bila histori sah
  lebih lama dibutuhkan. `unixtime` tidak digunakan untuk menimpa `date_time`.
- Duplikat waktu per perangkat memakai ID terakhir. Paket SM/ST/ph/SEC nol
  bersamaan ditandai mencurigakan dan dikeluarkan dari rata-rata (kebijakan dapat
  diubah dengan `exclude_zero_packet=False` setelah inspeksi perangkat).
  Nilai nol lainnya tetap dipertahankan. Celah tidak diinterpolasi.
- Ada enam model, masing-masing memakai 24 jam parameter itu sendiri untuk
  memprediksi parameter yang sama pada 24 jam berikutnya secara langsung.
  Kekosongan parameter lain tidak membuang jendela training parameter ini.
  Ridge dibandingkan dengan nilai terakhir.
  Pemilihan metode per sensor hanya menggunakan validation, lalu hasil dilaporkan
  pada test yang lebih baru. Target pelatihan berakhir sebelum blok berikutnya.
- Pembagian waktu kira-kira 60%/20%/20% berdasarkan waktu asal unik per parameter,
  dengan batas waktu global lintas device dan pemisahan horizon 24 jam.
  Data masa depan dari device lain tidak masuk blok training sebelumnya.
  MAE memakai satuan sensor asli. Titik awal evaluasi saling tumpang tindih;
  bukan percobaan independen. Tidak ada klaim interval kepercayaan atau akurasi
  kejadian kebakaran dari evaluasi sensor ini.
- `catatan_evaluasi` menandai sensor yang MAE test-nya lebih buruk dari baseline.
  Prediksi masih bersifat eksperimen; pemilihan di validation tidak menjamin
  kinerja yang lebih baik di test. Sensor konstan dicatat dalam audit, karena
  error nol pada sensor konstan bukan bukti bahwa alat/model bekerja baik.
- Model akhir dilatih ulang dengan semua jendela lengkap. Minimal 168 waktu asal unik dengan jendela
  lengkap per parameter adalah persyaratan teknis awal, bukan bukti kecukupan musiman.
  Keenam parameter harus memenuhi syarat training sebelum model ditulis.
  Tanpa 24 jam input terbaru lengkap, perangkat dilewati dengan alasan audit.
  Perangkat yang waktu inputnya mendahului batas akhir training juga dilewati
  agar model tidak menggunakan data masa depan pada prediksi historisnya.
- Data lebih dari dua jam dari asal prediksi ditandai `data_lama`. Forecast
  dimulai setelah jam lengkap terakhir, bukan otomatis 24 jam dari waktu saat ini.
  Perbarui telemetri untuk prediksi kondisi sekarang.

## Klasifikasi Normal, Rawan, Bahaya

Database saat diperiksa belum memiliki label status. Modul menyediakan klasifikasi
berbasis aturan melalui `aturan_risiko.json`, dalam kondisi **belum aktif**.
Tanpa label atau ambang terverifikasi, sistem tidak dapat menentukan tiga kelas
risiko yang dapat dipertanggungjawabkan. Tidak ada angka ambang operasional yang
dibuat secara otomatis, dan tidak ada probabilitas kebakaran yang dilaporkan.

Setelah pembimbing/pakar menentukan aturan, isi `source`, konfirmasikan satuan dan
referensi dengan `units_and_reference_confirmed`, lalu isi `rules`. Setiap aturan
membutuhkan `sensor`, `direction` (`low` jika makin kecil makin berisiko, `high`
jika makin besar makin berisiko), serta angka `rawan` dan `bahaya`.
Untuk `low`: `bahaya < rawan`; untuk `high`: `bahaya > rawan`.
Aktifkan `enabled` hanya setelah konfigurasi benar. Tingkat akhir adalah tingkat
tertinggi yang terpicu; Normal berarti semua aturan yang dikonfigurasi tidak
terpicu, bukan jaminan tidak terjadi kebakaran. Data lama tidak menghasilkan
status peringatan saat ini. `classify_rules` juga dapat dipanggil untuk observasi
aktual yang telah diberi pemeriksaan kesegaran.

Evaluasi klasifikasi membutuhkan label independen yang diverifikasi di lapangan
beserta waktu dan perangkatnya. Label dari aturan yang sama tidak dapat dipakai
untuk membuktikan akurasi terhadap kejadian kebakaran. Skema pelabelan yang
disarankan: `device_id,date_time,status,sumber_label`; status harus
Normal/Rawan/Bahaya. ph dan SEC tetap diprediksi, tetapi tidak otomatis menjadi
pemicu bahaya tanpa dasar yang tervalidasi.

## Hasil

`hasil_ews/prediksi_24_jam.csv`: 24 baris per perangkat yang lolos pemeriksaan,
enam sensor, asal prediksi, penanda data lama, status dan alasan.
`hasil_ews/evaluasi_mae.csv`: MAE per parameter/per horizon gabungan seluruh device
beserta pembanding; tidak lagi memiliki kolom `device_id`.
`hasil_ews/audit_perangkat.csv`: jumlah data, duplikat, paket mencurigakan,
kelengkapan dan alasan perangkat dilewati. Metode dan batas training ada pada
metadata masing-masing model parameter.
`hasil_ews/model_parameter_{sensor}.joblib`: satu model dan metadata untuk setiap
parameter, termasuk daftar device training; hanya muat file model yang dipercaya.
Hasil akan diperbarui saat skrip/notebook dijalankan ulang.

Pemeriksaan kode: `python -m unittest test_ews_gambut.py`.

Referensi metode:
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[evaluasi deret waktu](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).

## Perbaikan model dan uji historis

Training tetap per parameter lintas device. `parameter_models.py` menyediakan
Ridge dengan target perubahan dari nilai terakhir (alpha 10/100/1000),
ExtraTrees dengan target perubahan (leaf 5/20, seed 42), musiman 24 jam,
rata-rata 6 jam, dan nilai terakhir. Metode dipilih pada validation untuk blok
horizon 1-6, 7-12, 13-24 jam; perlu minimal 2% penurunan MAE validation agar
menggantikan nilai terakhir. Input 24 jam yang konstan memakai nilai terakhir.

Default `strategy="conservative"` hanya mengaktifkan kandidat pada SM dan ST;
ph/SEC/WL/RR memakai nilai terakhir. Kebijakan ini dipilih sesudah membandingkan
dua periode historis: RR dan SEC memburuk pada test terbaru, kenaikan WL kecil.
Angka benchmark adalah evaluasi historis yang ikut digunakan untuk kebijakan
rilis, **bukan estimasi independen pada data baru**. Tidak ada jaminan bahwa
semua device membaik. `strategy="legacy"` mereproduksi logika model sebelumnya;
`strategy="improved"` menguji kandidat pada keenam parameter.

Reproduksi dari snapshot hourly lokal tepercaya:

```powershell
python benchmark_models.py --snapshot uji_model/hourly_snapshot.pkl
python -m unittest test_ews_gambut.py test_api.py
```

Hasil detail: `uji_model/perbandingan_model.csv`, `evaluasi_kandidat.csv`,
`evaluasi_per_device.csv`, dan `pemilihan_validation.json`. MAE dilaporkan dalam
satuan parameter, bukan persentase akurasi. Kolom `mae_nonconstant_input`
menghitung MAE hanya pada origin dengan input 24 jam bervariasi, sehingga error
nol sensor konstan tidak menutupi kualitas prediksi. Penyebutnya ada di
`nonconstant_input_origins`. API juga mengembalikan `method_per_horizon`.

Referensi: [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
dan [evaluasi berurutan dengan gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## Versi pembanding PyTorch

Implementasi GRU terpisah tersedia di `ews_gambut_pytorch.py`.
Lihat [panduan PyTorch](README-pytorch.md) untuk instalasi, training dan
perbandingan MAE dengan model klasik pada snapshot serta split waktu yang sama.
Hasilnya disimpan di `hasil_ews_pytorch/`; API tetap memakai model klasik.

Untuk melatih setiap pasangan device-parameter secara terpisah lalu memilih
satu pemenang per parameter, lihat [panduan PyTorch per device](README-pytorch-device.md).
#   p u n y a g w k e r a s  
 