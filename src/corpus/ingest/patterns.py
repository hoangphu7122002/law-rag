"""Patterns of Vietnamese legal documents in the course corpus and vbpl (see docs/report/corpus-eda.md)."""
import re

REL_HEADERS = {
    "Van ban duoc can cu": "based_on", "Van ban duoc dan chieu": "cites", "Van ban huong dan": "guided_by",
    "Van ban duoc huong dan": "guides", "Van ban sua doi bo sung": "amended_by", "Van ban bi sua doi bo sung": "amends",
    "Van ban thay the": "replaced_by", "Van ban bi thay the": "replaces", "Van ban lien quan cung noi dung": "related",
    "Van ban lien quan ngon ngu": "translation",
}
REL_START = re.compile(r"^(" + "|".join(REL_HEADERS) + r"):\s*$")
REL_ITEM = re.compile(r"^\s*-\s*(.+?)\s*\(ID:\s*(\d+)\)\s*$")
TAIL = re.compile(r"-{3} (THÔNG TIN THÊM|MỤC LỤC|VĂN BẢN LIÊN QUAN) -{3}")
INFO_KEYS = {"Ten": "title", "So hieu": "number", "Loai van ban": "type", "Linh vuc nganh": "field",
             "Noi ban hanh": "issuer", "Ngay ban hanh": "issued", "Ngay hieu luc": "effective", "Tinh trang": "status"}

DOC_TYPES = ["VĂN BẢN HỢP NHẤT", "THÔNG TƯ LIÊN TỊCH", "THÔNG TƯ LIÊN BỘ", "NGHỊ QUYẾT LIÊN TỊCH", "BỘ LUẬT", "LUẬT",
             "PHÁP LỆNH", "NGHỊ ĐỊNH", "NGHỊ QUYẾT", "THÔNG TƯ", "QUYẾT ĐỊNH", "CHỈ THỊ", "KẾ HOẠCH", "THÔNG BÁO",
             "HƯỚNG DẪN", "CÔNG VĂN", "BÁO CÁO", "LỆNH", "ĐIỀU LỆ", "QUY ĐỊNH", "QUY CHẾ", "CHƯƠNG TRÌNH", "ĐỀ ÁN"]
TYPE_LINE = re.compile(r"^(" + "|".join(DOC_TYPES) + r")\b(.*)$")
BOILER = re.compile(  # whole-paragraph boilerplate, anywhere in the body
    r"^(CỘNG HOÀ|CỘNG HÒA|VIỆT NAM DÂN CHỦ CỘNG HÒA|VIỆT NAM DÂN CHỦ CỘNG HOÀ)\b.*$"
    r"|^Độc lập\s*[-–]\s*Tự do\s*[-–]\s*Hạnh phúc\.?$"
    r"|^[\s*_\-–=.~·•]+$"
    r"|^\(?Đã ký\)?\.?$|^\(Ký,.*\)$",
    re.IGNORECASE,
)
NUMBER = re.compile(r"^Số\s*:\s*(\S.*?)\s*$")
DATE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+((?:19|20)\d{2})")
NOI_NHAN = re.compile(r"^\s*Nơi nhận\s*:?", re.IGNORECASE)
# A paragraph that ends a "Nơi nhận" + signature block: attached rules / appendix / a new document starts.
RESUME = re.compile(r"^(Chương\s+[IVXLC\d]+|Điều\s+\d+|PHỤ LỤC|Phụ lục|Mẫu\s|MẪU\s|DANH MỤC|BIỂU|BẢNG|Mục\s+\d+"
                    r"|Phần\s+[IVX\d]+|PHẦN\s|(?:QUY ĐỊNH|QUY CHẾ|ĐIỀU LỆ|HƯỚNG DẪN|CHƯƠNG TRÌNH|KẾ HOẠCH|ĐỀ ÁN)\b)")
SIGN_TITLE = re.compile(r"^(TM\.|T/M|KT\.|K/T|Q\.|TL\.|TUQ\.|PP\.|CHỦ TỊCH|PHÓ CHỦ TỊCH|BỘ TRƯỞNG|THỨ TRƯỞNG|THỦ TƯỚNG"
                        r"|PHÓ THỦ TƯỚNG|GIÁM ĐỐC|PHÓ GIÁM ĐỐC|CHÁNH|TỔNG|THỐNG ĐỐC|PHÓ THỐNG ĐỐC|TRƯỞNG|CHỦ NHIỆM)")
SIGNATURE_MAX = 40   # paragraphs dropped at most after "Nơi nhận" when nothing resumes
HEADER_SCAN = 15     # paragraphs searched for the type line
LOCAL = re.compile(r"ỦY BAN NHÂN DÂN|UỶ BAN NHÂN DÂN|\bUBND\b|HỘI ĐỒNG NHÂN DÂN|\bHĐND\b|\bSỞ\b|\bTỈNH\b|THÀNH PHỐ"
                   r"|\bHUYỆN\b|\bQUẬN\b", re.IGNORECASE)
FORM_FILL = re.compile(r"[.…_]{5,}")   # blank-form placeholders "........" in attached templates
ISSUER_LINE = re.compile(r"^(CHỦ TỊCH|BỘ TRƯỞNG|THỦ TƯỚNG|GIÁM ĐỐC|THỐNG ĐỐC|TỔNG GIÁM ĐỐC|CHÁNH ÁN|VIỆN TRƯỞNG"
                         r"|ỦY BAN|UỶ BAN|HỘI ĐỒNG|CHÍNH PHỦ|QUỐC HỘI)\b")  # "CHỦ TỊCH UBND TỈNH X" above "Căn cứ"
