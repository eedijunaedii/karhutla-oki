# Sistem Peringatan Dini Karhutla Berbasis Machine Learning

Prediksi probabilitas titik panas (hotspot) **H+1 s.d. H+3** per sel grid ~5 km
untuk **Kabupaten Ogan Komering Ilir (OKI)**, menggunakan data iklim harian dan
model Random Forest / XGBoost dengan baseline Regresi Logistik.

> Tujuan: menggeser penanganan karhutla dari **reaktif** ke **preventif**.

---

## 1. Arsitektur pipeline

```
NASA FIRMS (titik panas)  ─┐
NASA POWER (cuaca harian)  ─┼─▶ grid 5km ─▶ join (sel × tanggal) ─▶ fitur lagged
                           │                                           │
                           │                                           ▼
                           │                              label fire_today + target H+k
                           │                                           │
                           ▼                                           ▼
                    data/processed/dataset ───────────▶ latih model ─▶ evaluasi
                                                                        │
                                                                        ▼
                                                        Dashboard Streamlit + Folium
```

| File | Fungsi |
|------|--------|
| `config.yaml` | Semua parameter (wilayah, grid, waktu, model). |
| `src/grid.py` | Bangun grid spasial ~5 km. |
| `src/download_firms.py` | Unduh + gabung titik panas FIRMS. |
| `src/download_power.py` | Unduh + gabung cuaca NASA POWER. |
| `src/features.py` | Fitur cuaca lagged (hari tanpa hujan, hujan kumulatif, lag). |
| `src/spatial_features.py` | Fitur spasial-temporal api (riwayat sel & tetangga, climatology). |
| `src/peat.py` | Ingest peta gambut KLHK/BRGM → fraksi gambut per sel (opsional). |
| `src/build_dataset.py` | Gabung semua → dataset latih berlabel. |
| `src/train.py` | Latih & evaluasi (LogReg / RF / XGBoost) per horizon. |
| `src/evaluate.py` | Metrik (Recall, Precision, PR-AUC) & kurva PR. |
| `app/dashboard.py` | Peta zona risiko Merah/Kuning/Hijau. |
| `scripts/make_sample_data.py` | Data **sintetis** untuk uji coba tanpa API. |
| `src/run_pipeline.py` | Orkestrasi end-to-end. |

---

## 2. Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

Paket inti (numpy, pandas, scikit-learn) wajib. `xgboost`, `imbalanced-learn`
(SMOTE), serta `streamlit`/`folium` bersifat opsional — pipeline tetap jalan
tanpa itu (model yang tak tersedia akan dilewati).

---

## 3. Coba cepat TANPA kredensial (data sintetis)

Untuk memastikan semuanya berjalan sebelum mengurus data asli:

```bash
python src/run_pipeline.py --sample
```

Ini membuat data iklim + titik panas sintetis (pola musim kemarau Sumatra),
membangun dataset, lalu melatih model. Lihat hasil di `reports/` dan `models/`.
Kemudian:

```bash
streamlit run app/dashboard.py
```

> ⚠️ Data sintetis hanya untuk menguji alur kode, **bukan** hasil ilmiah.

---

## 4. Menjalankan dengan DATA ASLI

### a. NASA FIRMS (titik panas) — butuh MAP_KEY gratis
1. Daftar MAP_KEY (instan): https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Isi `firms.map_key` di `config.yaml` (atau buat `config.local.yaml`).
3. ```bash
   python src/download_firms.py
   ```
   Arsip panjang (2015–2024) lewat API bisa lambat/terbatas. Alternatif:
   unduh manual dari https://firms.modaps.eosdis.nasa.gov/download/ lalu taruh
   CSV-nya di `data/raw/firms/` — skrip gabung akan membacanya otomatis.

### b. NASA POWER (cuaca) — tanpa API key
```bash
python src/download_power.py
```

### c. Bangun dataset + latih
```bash
python src/run_pipeline.py
```

---

## 5. Evaluasi

- Split **temporal**: latih `2015–2021`, uji `2022–2024` (atur di `config.yaml`).
- Imbalance ditangani via `class_weight` (default), opsi `smote: true`, atau
  `negative_undersample_ratio`.
- Metrik utama untuk kejadian langka: **Recall, Precision, PR-AUC** — tersimpan
  di `reports/metrics_h{k}.json` beserta kurva PR `reports/pr_h{k}_*.png`.
- Untuk poster IEEE: gunakan PR-AUC & recall sebagai metrik utama (akurasi
  menyesatkan pada data sangat tak seimbang), dan sertakan **keterbatasan**:
  resolusi POWER ~0.5°, titik panas satelit melewatkan api kecil/tertutup awan,
  bbox ≠ batas administrasi persis.

---

## 6. Fitur spasial & gambut

- **Fitur spasial-temporal** (`src/spatial_features.py`, otomatis aktif): riwayat
  api sel sendiri (7/30 hari), api sel tetangga (kemarin & 7 hari), hari sejak
  api terakhir, dan laju kebakaran historis sel (climatology). Semua dihitung
  hanya dari masa lalu (bebas kebocoran). **Ini menaikkan PR-AUC H+1 dari ~0,05
  ke ~0,23** pada data OKI 2015–2024 — kebakaran sangat terautokorelasi ruang-waktu.
- **Peta gambut** (`src/peat.py`, opsional): letakkan file gambut (`.shp`,
  `.geojson`, `.gpkg`, atau `.tif`) di `data/external/peat/` lalu jalankan ulang
  `python src/build_dataset.py` → kolom `peat_frac` ikut otomatis. Butuh
  `pip install geopandas shapely rasterio`.
  - **Sumber paling relevan (mencakup OKI):** portal gambut Sumsel
    `https://datagambut.sumselprov.go.id/` — layer **`geonode:lahan_gambut`**
    (sebaran gambut), `geonode:ketebalan_gambut`, `geonode:batas_khg`.
    Unduh otomatis: `python src/download_peat.py` (butuh server portal aktif).
  - **Resmi nasional:** BRGM (SIPALAGA), KLHK (Peta Fungsi Ekosistem Gambut),
    Indonesia Geospasial `tanahair.indonesia.go.id` (umumnya perlu akun).
  - **Alternatif global:** PEATMAP (University of Leeds) — shapefile global,
    potong ke extent OKI.

## 7. Pengembangan lanjutan (opsional)
- Tambah **CHIRPS** untuk presisi curah hujan.
- Kalibrasi probabilitas (Platt/Isotonic) sebelum ambang risiko.
- Validasi silang temporal (per tahun) untuk estimasi ketidakpastian.

---

## 8. Catatan penting
- `region` di `config.yaml` masih **bounding box perkiraan**. Untuk presisi,
  ganti dengan batas administrasi resmi OKI.
- Data mentah tidak di-commit (lihat `.gitignore`).
