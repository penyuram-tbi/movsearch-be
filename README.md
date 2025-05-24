# Movie Search (MOVSEARCH) API (Backend)

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
| **Pencarian Hybrid**| Menggabungkan pencarian kata-kunci dengan semantic dengan bobot yang bisa di adjust sendiri                                  |
| **Pencarian Kata Kunci** | Mendukung *multi-match* dengan fuzziness & boosting (*title*^3, *overview*^2, dll.)                         |
| **Filter Dinamis**    | Filter numerik (range) & kategorikal (term/terms) via parameter atau badan permintaan                          |
| **Skalabilitas**      | Elasticsearch sebagai *data store* + FastAPI untuk API asinkron                                               |
| **Index Re-Creation** | Skrip `scripts/index_data.py` mampu membuat ulang indeks & memasukkan data CSV                                |

---

## Arsitektur & Cara Kerja Sistem

### Backend (FastAPI & Elasticsearch)

#### Komponen Utama
- **FastAPI**: Menyediakan RESTful API yang skalabel dan asinkron
- **Elasticsearch**: Database dan mesin pencari yang menyimpan film beserta embedding vektor
- **Sentence Transformers**: Model bahasa yang mengubah teks menjadi representasi vektor (embeddings)

#### Alur Kerja Pencarian
1. **Preprocessing Data**:
   - Film dari dataset diproses menggunakan script `index_data.py`
   - Setiap film dikonversi menjadi dokumen Elasticsearch dengan embedding vektor menggunakan `sentence-transformers`
   - Metadata film (judul, deskripsi, genre, dll) disimpan sebagai field yang dapat dicari

2. **Pencarian Semantik**:
   - Ketika pengguna mengirim query, teks query diubah menjadi embedding vektor
   - Backend menghitung kemiripan kosinus antara vektor query dan vektor dokumen film
   - Film dengan skor kemiripan tertinggi dikembalikan sebagai hasil

3. **Pencarian Kata Kunci (Keyword)**:
   - Menggunakan fitur pencarian teks lengkap Elasticsearch (BM25)
   - Field penting seperti judul film diberi bobot lebih tinggi dalam pencarian

4. **Pencarian Hybrid**:
   - Menggabungkan skor dari pencarian semantik dan kata kunci
   - Parameter bobot dapat disesuaikan untuk menyeimbangkan kedua metode

5. **Pemfilteran Hasil**:
   - Pengguna dapat memfilter hasil berdasarkan tahun rilis, rating, genre, dll
   - Filter diterapkan setelah pencarian untuk mempersempit hasil

### Frontend (Next.js)

#### Komponen Utama
- **Next.js**: Framework React untuk rendering sisi klien dan server
- **Shadcn UI**: Komponen UI untuk antarmuka yang responsif dan modern
- **API Client**: Menangani komunikasi dengan backend

#### Alur Kerja Antarmuka
1. **Halaman Utama**:
   - Pencarian sederhana dengan opsi beralih antara mode pencarian
   - Filter tambahan untuk mempersempit hasil berdasarkan berbagai kriteria

2. **Halaman Hasil Pencarian**:
   - Menampilkan film yang sesuai dengan query
   - Card film dengan informasi dasar dan opsi untuk melihat detail

3. **Halaman Detail Film**:
   - Informasi lengkap tentang film yang dipilih
   - Metadata seperti tahun rilis, rating, genre, sinopsis, dll

4. **Fitur UI Lainnya**:
   - Pemfilteran dinamis berdasarkan berbagai kriteria
   - Toggle untuk beralih antara mode pencarian
   - Antarmuka responsif untuk berbagai ukuran layar

---

## Teknologi

- **Python ≥ 3.10**
- **FastAPI > 0.110**  
- **Elasticsearch (Server v8+)**  
- **sentence-transformers** (`all-MiniLM-L6-v2`, dimensi 384)  
- **Uvicorn** (ASGI server)
- **Next.js** (Frontend)
- **TypeScript** 
- **Shadcn UI Components** 

---

## Prasyarat

1. **Python** terinstal (direkomendasikan menggunakan *virtual environment*).
2. **Elasticsearch v8** sudah berjalan & memiliki API-Key dan ES-URL (mode *elastic-cloud* maupun self-hosted dengan `xpack.security` aktif).
3. File CSV yang akan di-indeks (contoh: `app/data/testSample.csv`).
4. **Node.js** dan **npm/pnpm** untuk frontend.

---

## Instalasi & Setup

```bash
# 1. Klon repositori
git clone https://github.com/penyuram-tbi/movsearch-be.git && cd movsearch-be

# 2. Buat virtual env
python -m venv venv

source venv/bin/activate (pada MAC-OS/Linux) 

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
INDEX_NAME=<your_index_name>

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

## Menjalankan Frontend
```bash
# 1. Pindah ke direktori frontend
cd ../movsearch-fe

# 2. Install dependencies
npm install
# atau
pnpm install

# 3. Jalankan server development
npm run dev
# atau
pnpm dev

# Akses http://localhost:3000 di browser
```

## REST API

| Metode | Endpoint                | Deskripsi                                                               |
| ------ | ----------------------- | ----------------------------------------------------------------------- |
| `POST` | `/api/v1/movies/semantic-search` | Pencarian **semantik**; badan permintaan mengikuti model `QueryRequest` |
| `POST` | `/api/v1/movies/hybrid-search` | Pencarian **hybrid**; badan permintaan mengikuti model `QueryRequest` dengan opsi pengaturan bobot |
| `POST` | `/api/v1/movies/keyword-search` | Pencarian **kata kunci**; badan permintaan mengikuti model `KeywordSearchRequest` |
| `GET`  | `/api/v1/movies/search` | Pencarian **kata kunci** + filter query-param                           |
| `GET`  | `/api/v1/movies/{id}`   | Ambil detail film berdasarkan ID                                        |

## Evaluasi & Optimasi Pencarian

Proyek ini menyertakan modul evaluasi di `app/evaluation/` untuk mengukur dan mengoptimalkan kualitas pencarian:

- **LLM Evaluator**: Menggunakan model bahasa untuk mengevaluasi relevansi hasil pencarian
- **Optimasi Bobot**: Pencarian bobot optimal antara pencarian semantik dan kata kunci
- **Metrik**: Metrik evaluasi standar termasuk precision, recall, dan F1-score

Hasil evaluasi tersimpan di direktori `evaluation_results/` untuk analisis lebih lanjut.