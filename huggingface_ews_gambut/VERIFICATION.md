# Pengujian paket

Paket disiapkan pada 15 November 2026. Pada 16 November 2026 paket disiapkan
untuk upload ke repository model privat `Cijay67/Nyawit`, bukan deployment Space.

Pemeriksaan yang sudah lulus:

- 10 tes unit HTTP, autentikasi, kontrak sumber HTTPS, penolakan redirect
  lintas host, pemeriksaan device, timestamp, nilai kosong dan histori pendek.
- Enam model `.keras` identik byte-per-byte dengan champion training sebelumnya.
- Semua checksum model sesuai metadata.
- Server Uvicorn berjalan dari folder ini sendiri, tanpa impor kode proyek induk.
- HTTP `/health`, `/docs`, `/openapi.json`, daftar model/device, latest,
  classification, results dan prediksi GET/POST berhasil pada mode example.
- 24 titik GET dan POST identik, serta cocok dengan prediksi Keras asli
  pada input contoh yang sama (toleransi numerik 1e-6).
- Bridge REST diuji terhadap MySQL lokal asli dengan Bearer token. Kontrak
  sumber -> paket inferensi -> hasil berhasil untuk M0014 dan M0015, masing-masing
  24 titik; paket nol M0013 ditolak dengan status eksplisit.
- Resolver pip untuk Linux x86_64/Python 3.12 berhasil menyelesaikan dependency
  runtime tanpa konflik. Pemeriksaan ini dry-run, bukan instalasi atau build image.
- File proyek lama tetap identik berdasarkan hash.

Batas pengujian:

- Docker tidak terpasang pada komputer pengujian, sehingga `docker build`
  belum dijalankan. Dockerfile masih harus dibuild oleh Hugging Face atau Docker lokal.
- Uji bridge menggunakan transport internal untuk kontrak HTTPS dengan database
  asli; belum menghubungi URL HTTPS sumber publik atau Cloudflare Tunnel nyata.
- Login HF berhasil sebagai Cijay67. Pembuatan Docker Space privat cpu-basic
  ditolak dengan HTTP 402: pesan server meminta langganan PRO untuk hosting
  Docker/Gradio. Tidak ada langganan yang dibeli atau Secrets Space yang dipasang.
  Repository model hanya menyimpan file; REST API belum berjalan di Hugging Face.
- Angka model berasal dari waktu sintetis, bukan bukti akurasi lapangan.

Setelah deploy, uji `/health`, `/api/v1/devices`, dan satu endpoint prediksi
melalui URL aplikasi Space untuk mengonfirmasi build, sertifikat HTTPS, token
dan jaringan sumber yang sebenarnya.
