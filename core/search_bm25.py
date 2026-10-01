"""BM25 ile kelime tabanlı arama (SQLite FTS5).

Kullanıcının yazdığı metin doğrudan FTS5'e verilemez: tırnak, parantez,
AND/OR/NOT gibi kelimeler FTS5'in sorgu dilinde özel anlam taşır ve hata verir.
Bu yüzden metni kelimelere ayırıp her kelimeyi tırnak içine alıyoruz ve
OR ile bağlıyoruz. Sıralamayı BM25 yapar: nadir kelimeler daha çok puan getirir.
"""
import re
import sqlite3

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"\w+", re.UNICODE)

_SQL = """
SELECT p.id, -f.rank AS score
FROM (SELECT rowid, rank FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?) AS f
JOIN papers p ON p.pk = f.rowid
ORDER BY f.rank
"""


def to_fts_query(text: str) -> str:
    """Serbest metni güvenli bir FTS5 sorgusuna çevirir. Kelime kalmazsa boş döner.

    "the", "of" gibi çok sık İngilizce kelimeler atılır: sıralamaya katkıları
    yok denecek kadar azdır ve 3 milyon makalede aramayı ciddi yavaşlatırlar.
    """
    words = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 1 and w not in ENGLISH_STOP_WORDS and w not in words:
            words.append(w)
    return " OR ".join(f'"{w}"' for w in words)


def search(conn: sqlite3.Connection, text: str, k: int = 10) -> list[tuple[str, float]]:
    """(makale kimliği, skor) listesi döner. Skor büyükse daha alakalı."""
    query = to_fts_query(text)
    if not query:
        return []
    return [(r[0], r[1]) for r in conn.execute(_SQL, (query, k))]
