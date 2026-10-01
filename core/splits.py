"""Makaleleri kalıcı olarak eğitim ve test havuzlarına ayırır.

Bir makalenin hangi havuza düştüğü sadece kimliğinden hesaplanır. Bu yüzden:
- Kod her çalıştığında aynı makale aynı havuza düşer.
- Her gün gelen yeni makaleler de kendiliğinden bir havuza yerleşir.
- Değerlendirme sorguları test havuzundan, ileride eğiteceğimiz modellerin
  verisi eğitim havuzundan üretilir. Böylece model sınavdaki soruları önceden görmez.
"""
import hashlib

TEST_PERCENT = 1  # makalelerin yüzde 1'i test havuzunda


def split_of(paper_id: str) -> str:
    bucket = int(hashlib.md5(paper_id.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if bucket < TEST_PERCENT else "train"
