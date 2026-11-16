# Hasil Keras pada waktu sintetis

Model per device-parameter. Satu pemenang per parameter dipilih dari macro MAE validation pada device pembanding yang sama; test tidak memilih pemenang.

| Parameter | Device pemenang | MAE validation | MAE test | Baseline test |
|---|---|---:|---:|---:|
| SM | LPA-C2413122025-M0015 | 3.990157 | 1.184064 | 1.376403 |
| ST | LPA-C2313122025-M0011 | 0.201268 | 0.228572 | 0.228540 |
| ph | LPA-C2613122025-M0014 | 0.032717 | 0.020729 | 0.020726 |
| SEC | LPA-C3113122025-M0019 | 0.391393 | 0.081690 | 0.079842 |
| WL | LPA-C3113122025-M0019 | 0.971737 | 0.621492 | 0.608609 |
| RR | LPA-C2413122025-M0015 | 0.207313 | 0.101654 | 0.127117 |

Data menggunakan waktu buatan 10 menit, bukan waktu pengukuran lapangan asli. MAE ini bukan bukti akurasi kebakaran atau perbaikan pada waktu asli.
Data 10 menit diagregasi menjadi rata-rata per jam, kemudian 24 jam input memprediksi 24 jam berikutnya. RR tetap rata-rata pembacaan, bukan total hujan.
Pembobotan macro sama untuk tiap device pembanding. Sensor training konstan dikecualikan dari ranking. Scaler hanya memakai bagian fit pada device asal.
Pemenang berarti terbaik dari kandidat layak; bukan jaminan mengalahkan baseline. Epoch dan kandidat dipilih dari validation yang sama, sehingga masih perlu konfirmasi pada data baru.
File lama, model lama, dan data database tidak diubah oleh training ini.
