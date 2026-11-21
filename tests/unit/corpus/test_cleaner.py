from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from corpus import DieuNodeParser, LawCleaner

COURSE = """ỦY BAN NHÂN DÂN
 TỈNH AN GIANG

-------

CỘNG HÒA XÃ HỘI
 CHỦ NGHĨA VIỆT NAM

Độc lập - Tự do - Hạnh phúc

---------------

Số: 06/2008/CT-UBND

An Giang, ngày 28 tháng 7 năm 2008

CHỈ THỊ

VỀ VIỆC TĂNG CƯỜNG HOẠT ĐỘNG
 CỦA TỔ CHỨC CÔNG ĐOÀN

Điều 1. Các ngành phải tạo điều
 kiện cho công đoàn.

Điều 2. Chỉ thị có hiệu lực sau 10 ngày.

Nơi nhận:

- Như Điều 2;

- Lưu: VT.

TM. ỦY BAN NHÂN DÂN

CHỦ TỊCH

Lâm Minh Chiếu

QUY CHẾ

PHỐI HỢP CÔNG TÁC

Điều 1. Nội dung phối hợp hằng quý.

GIÁM ĐỐC

Nguyễn Văn An--- THÔNG TIN THÊM ---
Ten: Chỉ thị 06/2008/CT-UBND tăng cường hoạt động công đoàn
So hieu: 06/2008/CT-UBND
Noi ban hanh: Tỉnh An Giang
Ngay ban hanh: 28/07/2008
Tinh trang: Còn hiệu lực


--- VĂN BẢN LIÊN QUAN ---

Van ban duoc dan chieu:
  - Luật Công đoàn 1990 (ID: 37890)
"""


def test_parse_course_file():
    d = LawCleaner.parse(COURSE)
    assert d.meta["title"] == "Chỉ thị 06/2008/CT-UBND tăng cường hoạt động công đoàn"
    assert d.meta["status"] == "Còn hiệu lực" and d.meta["level"] == "local" and d.meta["year"] == 2008
    assert d.relations == [{"rel": "cites", "title": "Luật Công đoàn 1990", "target_id": 37890}]
    lines = d.text.split("\n")
    assert lines[0] == "Điều 1. Các ngành phải tạo điều kiện cho công đoàn."       # header gone, wraps joined
    for noise in ("CỘNG HÒA", "Nơi nhận", "Lưu: VT", "Lâm Minh Chiếu", "Nguyễn Văn An", "GIÁM ĐỐC", "THÔNG TIN THÊM", "ID:"):
        assert noise not in d.text
    assert "QUY CHẾ" in lines and lines[-1] == "Điều 1. Nội dung phối hợp hằng quý."  # attached rules kept


def test_vbpl_first_line_is_title():
    raw = "Thông tư số 02/2013/TT-BNG Hướng dẫn cấp hộ chiếu\nTHÔNG TƯ\nHướng dẫn cấp hộ chiếu\n____\nCăn cứ Nghị định 136;\nĐiều 1. Phạm vi"
    d = LawCleaner.parse(raw, "vbpl")
    assert d.meta["title"].startswith("Thông tư số 02/2013/TT-BNG")
    assert d.text.split("\n") == ["Căn cứ Nghị định 136;", "Điều 1. Phạm vi"]


def test_web_folders_first_line_is_title():
    raw = "Thông tư số 02/2013/TT-BNG Hướng dẫn cấp hộ chiếu\nTHÔNG TƯ\nHướng dẫn cấp hộ chiếu\n____\nCăn cứ Nghị định 136;\nĐiều 1. Phạm vi"
    for src in ("web", "web_new_laws"):
        d = LawCleaner.parse(raw, src)
        assert d.meta["title"].startswith("Thông tư số 02/2013/TT-BNG"), src


def test_transform_title_in_embed_and_llm_text_only():
    doc = Document(text=COURSE, id_="x", metadata={"doc": "x", "source": "corpus"})
    [clean] = LawCleaner()([doc])
    nodes = DieuNodeParser(max_chars=2000)([clean])
    assert len(nodes) >= 3
    for mode in (MetadataMode.EMBED, MetadataMode.LLM):
        text = nodes[0].get_content(metadata_mode=mode)
        assert text.startswith("Chỉ thị 06/2008/CT-UBND tăng cường hoạt động công đoàn\nĐiều 1.")
        assert "Còn hiệu lực" not in text and "corpus" not in text
