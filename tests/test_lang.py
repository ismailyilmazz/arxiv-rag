import pytest

from core.lang import is_turkish


@pytest.mark.parametrize("text", [
    "yolo gerçek zamanlı nesne tespiti",
    "xgboost gradient boosting orijinal makale",
    "word2vec kelime embeddingleri mikolov",
    "Kahve nasıl demlenir?",
])
def test_turkish(text):
    assert is_turkish(text)


@pytest.mark.parametrize("text", [
    "Attention Is All You Need",
    "dense passage retrieval question answering",
    "How do I fix a leaky faucet?",
])
def test_english(text):
    assert not is_turkish(text)
