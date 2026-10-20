import sys

from arxiv_rag import config
from arxiv_rag.generation import gemini


def main() -> int:
    key = config.GEMINI_API_KEY
    if not key:
        print("GEMINI_API_KEY (ya da GEMINI_KEY) .env içinde bulunamadı.")
        return 1
    print(f"Anahtar bulundu: {key[:3]}... ({len(key)} karakter)")
    working = []
    for auth in ("bearer", "header"):
        try:
            models = gemini.list_models(auth)
            print(f"Kimlik doğrulama '{auth}': çalışıyor, {len(models)} model listelendi.")
            working.append((auth, models))
        except gemini.GeminiUnavailable as e:
            print(f"Kimlik doğrulama '{auth}': hata. {e}")
    if not working:
        print(
            "İki yöntem de çalışmadı. .env satırında tırnak ya da boşluk olmadığını kontrol et, gerekirse yeni anahtar al."
        )
        return 1
    auth, models = working[0]
    flash = [m for m in models if "flash" in m]
    print("Flash modelleri:", ", ".join(flash) or "yok")
    model = gemini.pick_model(models, config.GEMINI_MODEL)
    if model != config.GEMINI_MODEL:
        print(f"GEMINI_MODEL={config.GEMINI_MODEL} listede yok; deneme için {model or '(yok)'} kullanılıyor.")
    if not model:
        return 1
    try:
        text, tokens, finish = gemini.complete("Reply with the single word OK.", max_tokens=200, model=model, auth=auth)
        print(f"Deneme çağrısı: '{text[:40]}', {tokens} token, bitiş nedeni '{finish}'.")
    except gemini.GeminiUnavailable as e:
        print(f"Deneme çağrısı başarısız: {e}")
        return 1
    print("\n.env için öneri:")
    print(f"GEMINI_AUTH={auth}")
    print(f"GEMINI_MODEL={model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
