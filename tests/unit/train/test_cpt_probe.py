from train.cpt_probe import make_items, score

ART = {"key": "49/2005/QH11#13", "title": "Luật Các công cụ chuyển nhượng số 49/2005/QH11",
       "text": "Điều 13. Mất công cụ chuyển nhượng\n1. Khi công cụ bị mất, người thụ hưởng phải thông báo bằng văn bản.\n2. Thời hạn thông báo là 30 ngày kể từ ngày mất."}


def test_make_items_masks_a_number_and_keeps_the_law_and_article_in_the_prompt():
    items = make_items([ART], per_article=1, seed=0)
    assert len(items) == 1
    it = items[0]
    assert it["target"] == "30" and it["prompt"].endswith("là") and "Điều 13" in it["prompt"] and "49/2005/QH11" in it["prompt"]
    assert "30 ngày" not in it["prompt"] and it["key"] == ART["key"]


def test_make_items_skips_articles_without_a_usable_number_and_the_heading_number():
    assert make_items([{"key": "x#1", "title": "Luật X", "text": "Điều 1. Phạm vi\nQuy định chung."}], 2, 0) == []


def test_score_is_the_share_of_predictions_that_start_with_the_target_number():
    items = [{"target": "30"}, {"target": "5"}, {"target": "2"}]
    assert score(items, ["30 ngày", "5,", "3"]) == 2 / 3
