import pytest

from arxiv_rag.generation import llm
from scripts.generation import retranslate

EN = "Hallucinations in large language models jeopardize the trustworthiness of the text and raise concerns [1]."
TR = "Büyük dil modellerindeki halüsinasyonlar metnin güvenilirliğini tehlikeye atar ve kaygılara yol açar [1]."
REPORT = {
    "doc_type": "proposal",
    "draft": {
        "label": "research proposal on x",
        "lang": "tr",
        "title": "Başlık】",
        "names": {"Broad Surveys": "Broad Surveys"},
        "glossary": {},
        "sections": [
            {"key": "abstract", "heading": "Abstract", "heading_out": "Özet", "text": EN},
            {
                "key": "theme:Broad Surveys",
                "heading": "Related Work: Broad Surveys",
                "heading_out": "İlgili Çalışmalar: Broad Surveys",
                "text": EN,
            },
        ],
    },
}


def test_retranslate_rebuilds_headings_and_measures(monkeypatch):
    prompts = []

    def complete(prompt, model, max_tokens):
        prompts.append(prompt)
        if prompt.startswith("Translate each line below"):
            return "Broad Surveys", 10, "stop"
        if prompt.startswith("Translate this heading"):
            return "Genel Derlemeler", 10, "stop"
        return TR, 10, "stop"

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "chat", lambda prompt, model, **kw: ({"terms": []}, 5))
    markdown, summary = retranslate.retranslate(REPORT, "m")
    assert markdown == f"# Başlık\n\n## Özet\n\n{TR}\n\n## İlgili Çalışmalar: Genel Derlemeler\n\n{TR}\n"
    assert summary["headings"] == {"translated": 1, "of": 1, "from_plan": 0}
    assert summary["language"]["english_lines"] == 0 and summary["tokens"] == 5 + 20 + 20
    assert summary["glossary"] == {"Broad Surveys": "Genel Derlemeler"}


def test_report_without_draft_is_explained():
    with pytest.raises(ValueError, match="taslak"):
        retranslate.retranslate({"doc_type": "survey"}, "m")
