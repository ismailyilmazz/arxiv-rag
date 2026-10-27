# arXiv Araştırma Asistanı

[![CI](https://github.com/ismailyilmazz/arxiv-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ismailyilmazz/arxiv-rag/actions/workflows/ci.yml)

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

İsteğe bağlı: `LLM_CARD_MODEL`, `LLM_WRITE_MODEL`, `LLM_TPM`, `FULLTEXT_POLICY` (`all`/`cc`), `S2_API_KEY`,
`QUERY_TRANSLATION` (`off`/`replace`/`fuse`; Türkçe sorguları aramadan önce İngilizceye çevirir, önerilen `replace`),
`GEMINI_API_KEY` (Gemini katmanı; yoksa katman atlanır), `GEMINI_MODEL`, `GEMINI_AUTH` (`bearer`/`header`), `GEMINI_RPM`.
Gemini'nin canlıdaki durumu `/api/health` cevabındaki `gemini` alanında görünür.

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

## Yayın (Hugging Face Space)

`main`'e push edilen her commit CI'dan geçerse `Deploy` iş akışı arayüzü derler, kuyruk boşalınca Space'e yükler ve
`/api/health` cevabındaki `build.commit` o commit'i gösterene kadar bekler. Bunun için repo ayarlarında `HF_TOKEN` (Write)
gizli anahtarı gerekir. Elle yükleme yedek yol olarak duruyor:

```
cd frontend
npm run build
cd ..
python -m scripts.deploy.build_space --push <kullanıcı>/<space>
```

Space açılışta veritabanını, vektörleri ve modelleri veri deposundan `scripts.deploy.fetch_data` ile indirir.
Gemini katmanı için Space ayarlarına `GEMINI_API_KEY` secret'ı eklenir; durumu `/api/health` cevabındaki `gemini`
alanında görünür.

## Otomatik işler (GitHub Actions)

- `CI`: her push ve pull request'te, ayrıca her pazartesi; Python 3.11/3.12 testleri, lint ve arayüz.
- `Deploy`: `main`'de CI geçince Space'e yükler ve `/api/health`'te yeni commit'i bekler.
- `Smoke`: her pazartesi canlı siteyi yoklar ve LLM kotası harcamayan İngilizce bir arama yapar.
- `Data update`: her gün arXiv OAI-PMH'den (`https://oaipmh.arxiv.org/oai`) yeni ve güncellenen kayıtları çeker, gömer ve
  veri deposuna delta olarak yayımlar; Space açılışta delta'ları uygular. Çekme ya da kalite kontrolü başarısız olursa
  hiçbir şey yayımlanmaz ve son başarılı tarih ilerlemez. Bozuk bir delta Space'i durdurmaz, atlanır ve `/api/health`
  cevabında görünür. Gerekenler: `HF_TOKEN` ve `ARXIV_CONTACT_EMAIL` gizli anahtarları.

## Testler

```
pytest
ruff check .
ruff format --check .
cd frontend
npm test
```
