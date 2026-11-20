import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from demo import render


def reply(answer="Căn cứ Điều 5 ..."):
    return {"answer": answer, "sources": [{"doc": "27_2025_TT-NHNN", "text": "Điều 5. nội dung", "score": 0.987}],
            "timings": {"retrieve_s": 0.1, "rerank_s": 0.2, "generate_s": 2.5, "total_s": 2.8}}


def test_render_shows_question_answer_sources_and_latency():
    md = render(1, "Hạn nộp báo cáo là bao lâu?", reply())
    assert "### Q1." in md and "Hạn nộp báo cáo là bao lâu?" in md
    assert "Căn cứ Điều 5" in md and "27_2025_TT-NHNN" in md and "2.8 s" in md


def test_render_with_a_reference_adds_the_rouge_l_to_it():
    md = render(2, "q", reply("Căn cứ Điều 5 quy định thời hạn"), reference="Căn cứ Điều 5 quy định thời hạn")
    assert "ROUGE-L vs reference: 1.000" in md
    assert "ROUGE-L" not in render(3, "q", reply())
