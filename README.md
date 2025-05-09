

# Movie Search API (Backend)

Repositori ini berisi _backend_ untuk **Movie Search Engine** yang memanfaatkan **Elasticsearch** dan model *sentence-transformers* guna menyediakan pencarian semantik maupun pencarian berbasis kata kunci.

Contributor :
 - Akmal Ramadhan - 2206081534
 - Muh.Kemal Lathif Galih Putra - 2206081225
 - Tsabit Coda Rafisukmawan - 2206081414

Repositori ini dibuat sebagai bagian dari proyek akhir mata kuliah Temu Balik Informasi di Universitas Indonesia.


## Fitur Utama

| Fitur                 | Deskripsi                                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------------------------|
| **Pencarian Semantik**| Menghitung kemiripan kosinus antara embedding query & dokumen (dense vector)                                  |
| **Pencarian Kata Kunci** | Mendukung *multi-match* dengan fuzziness & boosting (*title*^3, *overview*^2, dll.)                         |
| **Filter Dinamis**    | Filter numerik (range) & kategorikal (term/terms) via parameter atau badan permintaan                          |
| **Skalabilitas**      | Elasticsearch sebagai *data store* + FastAPI untuk API asinkron                                               |
| **Index Re-Creation** | Skrip `scripts/index_data.py` mampu membuat ulang indeks & memasukkan data CSV                                |

---

## Teknologi

- **Python ≥ 3.10**
- **FastAPI > 0.110**  
- **Elasticsearch (Server v8+)**  
- **sentence-transformers** (`all-MiniLM-L6-v2`, dimensi 384)  
- **Uvicorn** (ASGI server)

---

## Prasyarat

1. **Python** terinstal (direkomendasikan menggunakan *virtual environment*).
2. **Elasticsearch v8** sudah berjalan & memiliki API-Key (mode *elastic-cloud* maupun self-hosted dengan `xpack.security` aktif).
3. File CSV yang akan di-indeks (contoh: `app/data/testSample.csv`).

---

## Instalasi & Setup

```bash
# 1. Klon repositori
git clone https://github.com/username/movsearch-be.git && cd movsearch-be

# 2. Buat virtual env
python -m venv venv

source venv/bin/activate (pada OS) 

OR 

venv\Scripts\activate (pada windows)

# 3. Install semua hal yang diperlukan
pip install -r requirements.txt
```
## Pembuatan `.env`

1. Buat file `.env` pada root project (pastikan penggunaan "." pada awal file).

2. Isi file tersebut dengan code berikut :

``` bash
API_V1_STR=/api/v1
PROJECT_NAME=Movie Search API

# Elasticsearch configuration
ELASTICSEARCH_URL=https://<cluster-id>.es.<region>.aws.cloud.es.io:443
ELASTICSEARCH_API_KEY=<base64-api-key>
INDEX_NAME=semantic_documents  # opsional, default sama

# Vector model configuration
VECTOR_MODEL_NAME=all-MiniLM-L6-v2
VECTOR_DIMENSIONS=384
```

## Inisialisasi Indeks & Pengindeksan Data

Jalankan code berikut untuk melakukan indexing data:

``` bash
python scripts/index_data.py --csv app/data/testSample.csv --recreate
```
Argumen --recreate akan menghapus indeks lama & membuat ulang struktur mapping sebelum memasukkan dokumen baru.

## Menjalankan Server
```bash
uvicorn app.main:app --reload    # Akses http://127.0.0.1:8000
```

## REST API

| Metode | Endpoint                | Deskripsi                                                               |
| ------ | ----------------------- | ----------------------------------------------------------------------- |
| `POST` | `/api/v1/movies/search` | Pencarian **semantik**; badan permintaan mengikuti model `QueryRequest` |
| `GET`  | `/api/v1/movies/search` | Pencarian **kata kunci** + filter query-param                           |
| `GET`  | `/api/v1/movies/{id}`   | Ambil detail film berdasarkan ID                                        |




