# movsearch-be
Backend for Movie Search Engine

# How To Jalankan Apps ini
1. Pertama install dependency yang diperlukan untuk menjalankan apps ini
    ```
    pip install -r requirements.txt
    ```
2. Buat env file untuk:

    ```
    ELASTICSEARCH_URL = 
    ELASTICSEARCH_API_KEY = 
    INDEX_NAME = 
    ```

3. Jalankan apps ini dengan command
    ```
    uvicorn main:app --reload
    ```

3. Voilah apps sudah berjalan
