# Pemenang model per parameter

Setiap kandidat dilatih hanya pada satu device. Pemenang dipilih dari MAE validation pada data pembanding lintas device yang sama, dengan bobot sama per device.

Sensor dengan target training konstan tidak ikut ranking. Data test tidak dipakai memilih pemenang. Hasil tetap berupa eksperimen historis.

| Parameter | Device asal model | MAE validation macro | MAE test macro | Baseline test |
|---|---|---:|---:|---:|
| SM | LPA-C2413122025-M0015 | 4.356091 | 4.185713 | 4.653969 |
| ST | LPA-C2313122025-M0011 | 0.256435 | 0.310672 | 0.314740 |
| ph | LPA-C2613122025-M0014 | 0.054546 | 0.063793 | 0.063785 |
| SEC | LPA-C3113122025-M0019 | 0.537593 | 0.103985 | 0.105507 |
| WL | LPA-C2413122025-M0015 | 1.649137 | 1.961012 | 1.918675 |
| RR | LPA-C2613122025-M0014 | 0.163249 | 0.275639 | 0.276272 |

MAE mengikuti satuan masing-masing parameter. Nilai antarparameter tidak boleh dibandingkan langsung.
Pemenang berarti terbaik di antara kandidat yang lolos syarat, bukan jaminan mengungguli baseline atau model gabungan sebelumnya.
Model terbaik pada device asal saja belum tentu terbaik lintas device. Karena itu ranking utama memakai data validation pembanding yang sama; MAE device asal tetap disimpan terpisah.
Jumlah origin per device dapat berbeda. MAE macro memberi bobot sama pada setiap device pembanding, bukan pada jumlah barisnya.
Device pembanding ditentukan dari train dan validation yang cukup serta perubahan target training; daftar ini dikunci sebelum membaca test. Device tanpa origin test tidak berkontribusi ke skor test dan dicatat pada JSON evaluasi.
Checkpoint validation, evaluasi dan final dipisah. Hanya checkpoint sebelum test yang menghitung MAE test; checkpoint final menerima seluruh histori device asal.
Pemilihan epoch dan pemenang sama-sama memakai validation; konfirmasi pada data masa depan baru masih diperlukan. Origin yang tumpang tindih bukan percobaan independen.

Lihat ranking_model.csv, audit_device_parameter.csv, evaluasi_horizon.csv dan devices/{device_id}/evaluasi_{sensor}.json untuk rincian.
