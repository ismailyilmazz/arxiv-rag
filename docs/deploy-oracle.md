# Oracle Cloud'a yayına alma

Uygulama, Oracle Cloud'un ücretsiz ("Always Free") ARM sanal makinesinde Docker ile çalışır. Büyük veri
(veritabanı, vektörler, modeller) Kaggle'dan Hugging Face veri deposuna, oradan sunucuya gider; senin
bilgisayarından geçmez. Caddy, HTTPS sertifikasını kendisi alır. Adres, sunucunun IP'sinden üretilen
ücretsiz bir `sslip.io` adıdır (örneğin `130-61-12-34.sslip.io`).

```
Kaggle (04, 06) ─► notebook 10 ─► Hugging Face: ismailylmz283/arxiv-rag-data
                                         │
Oracle sanal makine ◄── fetch_data ──────┘
  docker compose: app (FastAPI + arayüz, 8000)  ◄──  caddy (80, 443, HTTPS)
```

## Gerekenler (hepsi en başta)

| Ne | Nereden | Not |
|---|---|---|
| Oracle Cloud hesabı | cloud.oracle.com | Kayıtta kimlik doğrulaması için kredi kartı ister. Always Free kaynaklar ücretsizdir. |
| Hugging Face yazma token'ı | huggingface.co → Settings → Access Tokens → New token, tür **Write** | Sadece Kaggle'da, veri yüklemek için. |
| Groq API anahtarı | Zaten var (`LLM_API_KEY`) | Sunucunun `.env` dosyasına girer. |
| SSH anahtarı | Bilgisayarında üretilir (Bölüm 2) | Sunucuya bağlanmak için. |

## Bölüm 1: Veriyi Hugging Face'e yükle (bir kez)

1. Kaggle'da yeni bir notebook aç ve `notebooks/10_publish_data.py` içeriğini tek hücreye yapıştır.
2. **Add Input** ile 04 (`04-full-embedding`) ve 06 (`06-train-models`) notebook'larının çıktılarını ekle.
3. Ayarlarda **Internet**'i aç. **Add-ons → Secrets** altında `HF_TOKEN` adıyla yazma token'ını ekle ve notebook'a bağla.
4. **Save & Run All**. Çıktıda modellerin scikit-learn sürümü `1.6.1` görünmeli. Son satır
   "Uzak depodaki boyutlar manifestle aynı." olmalı.

## Bölüm 2: Oracle sanal makinesi

1. Oracle Cloud'a kaydol. **Home region** (ana bölge) sonradan değiştirilemez ve Always Free ARM makineler sadece
   ana bölgede açılabilir.
2. Bilgisayarında PowerShell'de SSH anahtarı üret:
   ```
   ssh-keygen -t ed25519 -f $HOME\.ssh\oracle
   Get-Content $HOME\.ssh\oracle.pub
   ```
   İkinci komutun çıktısı (tek satır) açık anahtardır.
3. **Compute → Instances → Create instance**:
   - **Image:** Canonical Ubuntu 24.04.
   - **Shape:** Ampere → `VM.Standard.A1.Flex`, **4 OCPU, 24 GB** bellek.
   - **Networking:** yeni bir VCN ve public subnet; "Assign a public IPv4 address" seçili.
   - **Add SSH keys:** "Paste public keys" ve 2. adımdaki satırı yapıştır.
   - **Boot volume:** "Specify a custom boot volume size" ile **100 GB**.
   - "Out of capacity" hatası alırsan başka bir **availability domain** seç ya da daha sonra yeniden dene.
4. Makine açılınca **Instance details → Subnet → Security List → Add Ingress Rules** ile iki kural ekle:
   Source CIDR `0.0.0.0/0`, IP Protocol TCP, Destination Port `80`; aynısı `443` için.
5. Makinenin **Public IP** adresini not et. Site adresin, IP'deki noktaların tireye çevrilmiş hali ve sonuna
   `.sslip.io`: `130.61.12.34` → `130-61-12-34.sslip.io`.
6. **Hesabı Pay As You Go'ya yükselt** (Billing → Upgrade). Oracle, uzun süre çok az kullanılan Always Free
   makineleri geri alabiliyor; demo uygulaması çoğu zaman boşta durduğu için bu risk gerçek. Yükseltmeden sonra
   Always Free kaynaklar ücretsiz kalır, geri alma kuralı uygulanmaz. Ücretli bir kaynak açmadıkça fatura çıkmaz.

## Bölüm 3: Sunucuyu kur

1. Bağlan (IP'yi kendininkiyle değiştir):
   ```
   ssh -i $HOME\.ssh\oracle ubuntu@130.61.12.34
   ```
2. Repoyu indir ve kurulum betiğini çalıştır:
   ```
   git clone https://github.com/ismailyilmazz/arxiv-rag.git
   cd arxiv-rag
   bash deploy/setup.sh
   exit
   ```
   Betik Docker'ı kurar, güvenlik duvarında 80 ve 443'ü açar, `data/` klasörünü ve `.env` dosyasını hazırlar.
3. Yeniden bağlan (Docker yetkisi yeni oturumda geçerli olur) ve `.env`'i doldur:
   ```
   ssh -i $HOME\.ssh\oracle ubuntu@130.61.12.34
   cd arxiv-rag
   nano .env
   ```
   `LLM_API_KEY`, `ARXIV_CONTACT_EMAIL` ve `SITE_ADDRESS` (örneğin `130-61-12-34.sslip.io`) satırlarını doldur.
   `Ctrl+O`, `Enter`, `Ctrl+X` ile kaydet ve çık.
4. İmajı derle, veriyi indir, başlat:
   ```
   docker compose build
   docker compose run --rm app python -m scripts.deploy.fetch_data
   docker compose up -d
   ```
   `fetch_data` sonunda "N dosya, X GB, /data altında hazır." yazar. Yarıda kalırsa aynı komut kaldığı yerden devam eder.
5. Açılışı izle:
   ```
   docker compose logs -f app
   ```
   "Application startup complete" satırından sonra arama akışı arka planda belleğe yüklenir. `Ctrl+C` ile çık.
6. Tarayıcıda `https://<SITE_ADDRESS>` adresini aç.

## Bölüm 4: Bakım

| İş | Komut (`~/arxiv-rag` içinde) |
|---|---|
| Kodu güncelle | `git pull && docker compose up -d --build` |
| Durum | `docker compose ps` (app satırında `healthy` görünmeli) |
| Loglar | `docker compose logs --tail 200 app` |
| Veriyi doğrula | `docker compose run --rm app python -m scripts.deploy.fetch_data --check` |
| Yeniden başlat | `docker compose restart app` |
| Disk | `df -h` |

## Sorun giderme

- **Site açılmıyor:** Security List'teki 80 ve 443 kurallarını kontrol et; sunucuda `sudo iptables -L INPUT -n --line-numbers | head`
  çıktısında bu portlar için `ACCEPT` satırları olmalı. Sertifika hataları için `docker compose logs caddy`.
- **torch kurulumu başarısız:** Dockerfile, torch'u PyTorch'un CPU deposundan kurar (NVIDIA kütüphaneleri inmesin
  diye). O sürüm depoda yoksa Dockerfile'daki `torch==2.14.0` satırını depodaki en yakın sürümle değiştir.
- **"Veri eksik ya da bozuk":** `fetch_data`'yı yeniden çalıştır; eksik dosyaları tamamlar.
- **Bellek:** `docker stats --no-stream` ile izle. Tam korpusla vektörler bellekte yaklaşık 4,9 GB tutar.
