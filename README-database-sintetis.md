# Salinan database dengan interval 10 menit

Database baru: `tugasakhir_sintetis_10menit`.
Sumber: `tugasakhir.sensor_data`.

Salinan berisi semua baris dan semua kolom sumber. Waktu baru adalah **waktu
sintetis untuk eksperimen**, bukan pemulihan waktu pengukuran asli. Meratakan
jarak waktu mengasumsikan setiap baris pada suatu device mewakili 10 menit;
asumsi itu tidak membuktikan bahwa laju sampling asli memang 10 menit.

## Aturan urutan dan waktu

- Urutan mengikuti `id` asli dari kecil ke besar.
- `id` baru adalah primary key unik lintas device, mulai dari 0 hingga 143994
  untuk salinan pertama yang berisi 143995 baris.
- `device_sequence` mulai dari **0 untuk setiap device**, terpisah dari primary key.
- Semua device mulai pada `2026-01-01 00:00:00 WIB`.
- `date_time = waktu awal + device_sequence * 10 menit`.
- `unixtime` adalah Unix epoch dalam detik untuk waktu yang sama, dengan
  konversi zona waktu yang benar. Waktu WIB disimpan pada `date_time`.
- Nilai sensor tidak diubah, termasuk nol dan nilai kosong.

Misalnya, tiga pembacaan pertama pada masing-masing device mempunyai
`device_sequence` 0, 1, 2 dan waktu 00:00, 00:10, 00:20. Kedatangan baris device
lain tidak menaikkan nomor urut atau waktu device tersebut.

Kolom tambahan:

| Kolom | Arti |
|---|---|
| `device_sequence` | Nomor urut khusus device, mulai 0 |
| `original_id` | ID asli; unik dan tidak diubah |
| `original_date_time` | Nilai `date_time` asli, sebelum rekonstruksi |
| `original_unixtime` | Nilai `unixtime` asli |
| `time_is_synthetic` | Selalu 1 pada salinan ini |

`created_at` tetap waktu penerimaan asli, bukan waktu sintetis. Kolom bertipe
TIMESTAMP ini dibaca dan ditulis melalui session UTC agar waktu yang tersimpan
tidak bergeser. Tabel `dataset_metadata` menyimpan asal data, konfigurasi,
fingerprint sumber dan hasil pemeriksaan salinan.

## Memeriksa di MySQL

```sql
SELECT id, device_id, device_sequence, date_time, unixtime, original_id
FROM tugasakhir_sintetis_10menit.sensor_data
ORDER BY device_id, device_sequence
LIMIT 30;

SELECT device_id, COUNT(*) AS jumlah,
       MIN(device_sequence) AS urutan_awal,
       MAX(device_sequence) AS urutan_akhir,
       MIN(date_time) AS waktu_awal,
       MAX(date_time) AS waktu_akhir
FROM tugasakhir_sintetis_10menit.sensor_data
GROUP BY device_id;
```

Untuk mengarahkan pembacaan aplikasi ke salinan baru:

```powershell
$env:MYSQL_URL = 'mysql+pymysql://root:@localhost:3306/tugasakhir_sintetis_10menit'
$env:DB_TIMEZONE = 'Asia/Jakarta'
```

Jangan gunakan keluaran eksperimen waktu sintetis sebagai bukti akurasi
prediksi kondisi lapangan. Model yang dilatih pada waktu asli dan waktu
sintetis juga perlu dibedakan saat menyimpan dan mengevaluasi hasil.

## Skrip dan verifikasi

`salin_database_sintetis.py` secara default hanya membuat preview. Opsi
`--write` membuat database baru lalu menulis salinan. Nama database yang sudah
ada ditolak, dan sumber tidak ditimpa.

```powershell
python salin_database_sintetis.py --target nama_database_baru
python salin_database_sintetis.py --target nama_database_baru --write
python -m unittest test_salin_database_sintetis.py
```

Preview dan hasil verifikasi tersimpan di `hasil_salin_sintetis/`:
`preview.csv`, `ringkasan_per_device.csv`, dan `hasil.json`.
Pemeriksaan meliputi jumlah baris, kesetaraan semua kolom sumber, ID global
mulai 0, nomor urut per device, interval tepat 600 detik dan kesesuaian
`date_time`/`unixtime` pada zona waktu WIB. Data dibaca ulang setelah penulisan;
insert dibatalkan bila verifikasi dalam transaksi gagal.
