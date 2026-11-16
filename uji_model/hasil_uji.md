# Hasil uji historis model EWS gambut

Perbaikan diterapkan pada SM dan ST. Parameter lainnya memakai nilai terakhir.

MAE rata-rata horizon 1-24 jam pada origin test yang sama, dalam satuan parameter.

| Parameter | Model lama | Model diterapkan | Baseline | Penurunan MAE dari lama |
|---|---:|---:|---:|---:|
| RR | 0.275083 | 0.275083 | 0.275083 | 0.00% |
| SEC | 0.076877 | 0.076877 | 0.076877 | 0.00% |
| SM | 0.775662 | 0.752745 | 0.775662 | 2.95% |
| ST | 0.252737 | 0.218477 | 0.237231 | 13.56% |
| WL | 2.070039 | 2.070039 | 2.070039 | 0.00% |
| ph | 0.050588 | 0.050588 | 0.050588 | 0.00% |

## Batas kesimpulan

- Kebijakan SM/ST dipilih setelah membaca benchmark historis. Angka ini bukan evaluasi independen pada data baru.
- SM hanya bervariasi pada satu device yang berkontribusi ke test terbaru. MAE gabungan rendah karena device SM lain konstan.
- ST membaik secara gabungan, tetapi beberapa device memburuk. Lihat evaluasi_per_device.csv.
- Pada test terbaru, kandidat semua parameter menaikkan MAE RR 12,64% dan SEC 5,51%; keduanya tidak diterapkan. Perbaikan WL hanya 0,80%, sehingga tetap baseline.
- Sampel test saling tumpang tindih. Tidak dihitung interval kepercayaan atau akurasi kebakaran.
- Histori dengan jendela training lengkap berakhir 25 Mei 2026. Prediksi yang dihasilkan tetap historis/data lama.

## Reproduksi

`python benchmark_models.py --snapshot uji_model/hourly_snapshot.pkl`

Snapshot berasal dari pembacaan MySQL read-only, dibersihkan dengan prepare_device. Lag dan target disusun per device sebelum digabung.
Kandidat dipilih menggunakan validation, dengan gap target 24 jam sebelum validation dan test. Dua cutoff historis diuji tanpa tuning kandidat setelah melihat test.
Pemilihan metode per blok horizon memakai MAE validation dan syarat perbaikan 2%. Kandidat pohon menggunakan seed 42.
Rincian batas waktu, jumlah jendela, metode dan MAE validation tersimpan di pemilihan_validation.json.

## Input yang bervariasi

MAE di bawah hanya memakai origin dengan 24 jam input yang tidak konstan; penyebutnya bukan semua origin test.

| Parameter | Origin bervariasi | Total origin | MAE input bervariasi | Baseline input bervariasi |
|---|---:|---:|---:|---:|
| RR | 196 | 396 | 0.399081 | 0.399081 |
| SEC | 107 | 396 | 0.199377 | 0.199377 |
| SM | 66 | 396 | 4.516472 | 4.653969 |
| ST | 163 | 396 | 0.436756 | 0.482316 |
| WL | 383 | 396 | 2.094218 | 2.094218 |
| ph | 273 | 396 | 0.068244 | 0.068244 |

Referensi implementasi: [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html), [evaluasi waktu](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
