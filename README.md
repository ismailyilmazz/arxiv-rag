# arXiv Araştırma Asistanı

Bir konu yazarsın (Türkçe ya da İngilizce). Sistem 3,18 milyon arXiv makalesi içinde anlamsal arama, BM25, öğrenen
sıralayıcı ve bekçi modeliyle ilgili makaleleri bulur; 30 kaynaklık bir plan çıkarır, sen onaylarsın, sonra seçtiğin
türde (literatür taraması, araştırma önerisi, sentez raporu) her iddiası kaynağına bağlı bir metin yazılır.

## Kurulum

Gerekenler: Python 3.11 ya da üstü, Node.js LTS.

```
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev,embed]"
cd frontend
npm ci
npm run build
cd ..
```

Proje kökünde bir `.env` dosyası oluştur:

```
LLM_API_KEY=
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_MODEL_FAST=openai/gpt-oss-20b
ARXIV_CONTACT_EMAIL=
```

İsteğe bağlı: `LLM_CARD_MODEL`, `LLM_WRITE_MODEL`, `LLM_TPM`, `FULLTEXT_POLICY` (`all`/`cc`), `S2_API_KEY`.

Yerel veri (30 binlik örnek, vektörler ve modeller) `notebooks/08_local_bundle.py` çıktısındaki pakettir:

```
python -m scripts.data.import_bundle <yol>\local_bundle.zip
```

## Çalıştırma

```
uvicorn arxiv_rag.api.app:create_app --factory --port 8000
```

Arayüz `http://127.0.0.1:8000`, API belgeleri `http://127.0.0.1:8000/docs` adresindedir.

Arayüz üzerinde çalışırken API açıkken `cd frontend` ve `npm run dev`: Vite `http://127.0.0.1:5173` adresinde açılır ve
`/api` isteklerini 8000 portuna yönlendirir.

Komut satırından belge üretimi (tam korpus gerekir):

```
python -m scripts.generation.generate --topic "Mixture of experts for large language models" --type survey --dry-run
```

## Sunucuda çalıştırma (Docker)

Gerekenler: Docker ve Docker Compose, en az 16 GB bellek, 80 ve 443 portları açık bir sunucu.

```
cp deploy/env.example .env
docker compose build
docker compose run --rm app python -m scripts.deploy.fetch_data
docker compose up -d
```

`.env` içinde `SITE_ADDRESS` sitenin adresidir; Caddy bu adres için HTTPS sertifikasını kendisi alır. `fetch_data`
veritabanını, vektörleri ve modelleri `HF_DATA_REPO` veri deposundan indirip boyutlarını doğrular.

## Testler

```
pytest
ruff check .
ruff format --check .
cd frontend
npm test
```
