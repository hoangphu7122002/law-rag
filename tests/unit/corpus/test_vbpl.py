from corpus.vbpl import VbplImporter


def test_key_matches_course_numbers():
    assert VbplImporter.key("22/2023/QĐ-TTg") == VbplImporter.key("22/2023/QĐ-TT")
    assert VbplImporter.key(" 52/2024/NĐ-CP ") == "52/2024/NĐ-CP"


def test_filename_and_text():
    assert VbplImporter.filename("52/2024/NĐ-CP") == "52_2024_NĐ-CP"
    page = "<html><head><title>x</title></head><body><p>Điều 1.&nbsp;Phạm vi</p><p>1. Nội dung</p></body></html>"
    assert VbplImporter.to_text(page) == "Điều 1. Phạm vi\n1. Nội dung"
