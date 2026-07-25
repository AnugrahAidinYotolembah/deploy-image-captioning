# Streamlit Deploy - Pakaian Adat Captioning

Folder ini berdiri sendiri untuk deploy Streamlit. Folder lama di luar folder ini tidak perlu diubah.

## Isi penting

- `app.py`: aplikasi Streamlit.
- `model_utils.py`: loader model dan pipeline captioning/zero-shot/late-fusion.
- `models/model_pakaian_adat_bundle.pt`: model bundle baru untuk deploy.
- `scripts/build_model_bundle.py`: membuat bundle dari `../model_captioning_fast.pth` dan CSV dataset.
- `scripts/copy_segment_visualizations.py`: opsional, copy file visualisasi SAM3 saja ke `assets/output_segmentasi`.
- `sample_images/`: gambar contoh untuk uji cepat.

## Membuat model bundle

Dari folder ini:

```bash
/mnt/nas-hpg9/didiyotolembah/anaconda3/envs/ai_env/bin/python scripts/build_model_bundle.py
```

Bundle yang dibuat:

```text
models/model_pakaian_adat_bundle.pt
```

Bundle ini berisi:

- `image_encoder.fc`
- `caption_decoder`
- metadata dataset dari CSV
- vocab tokenizer BERT

Bundle tidak menyimpan optimizer, text encoder, dan bobot CLIP visual pretrained agar ukuran deploy lebih ringan. Saat app berjalan, CLIP pretrained tetap diload melalui library `clip`.

## Menyalin hasil segmentasi

Jika ingin semua visualisasi segmentasi tersedia di app:

```bash
/mnt/nas-hpg9/didiyotolembah/anaconda3/envs/ai_env/bin/python scripts/copy_segment_visualizations.py
```

Catatan: visualisasi segmentasi bisa berukuran ratusan MB. Untuk Streamlit Cloud, lebih aman memakai storage eksternal atau hanya membawa sample.

## Menjalankan lokal

`ai_env` saat dibuat belum memiliki Streamlit. Install dulu:

```bash
/mnt/nas-hpg9/didiyotolembah/anaconda3/envs/ai_env/bin/python -m pip install -r requirements.txt
```

Lalu jalankan:

```bash
/mnt/nas-hpg9/didiyotolembah/anaconda3/envs/ai_env/bin/streamlit run app.py
```

## Deploy publik

Untuk deploy publik, upload isi folder ini ke repo/aplikasi Streamlit. Karena model dan aset bisa besar, rekomendasi praktis:

- Simpan `model_pakaian_adat_bundle.pt` memakai Git LFS atau external object storage.
- Simpan `assets/output_segmentasi` hanya jika ukuran hosting memungkinkan.
- Jika memakai external storage, download file ke `models/` sebelum `app.py` dijalankan atau mount foldernya ke path yang sama.
