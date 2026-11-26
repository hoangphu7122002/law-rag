"""Fetch the cited laws that the store lacks (data/processed/missing_laws.json, written by the coverage check) from luatvietnam.vn.

    PYTHONPATH=src python scripts/fetch_missing_laws.py [--dry-run]
Refs come from Citations().table() whose law key is in the missing set; the existing polite crawler does the fetching and the
manifest is merged, not replaced. Rebuild / extend the index afterwards (a new document is invisible to a cached index).
"""
import argparse
import json

from corpus.keys import law_key
from corpus.sources.citations import Citations
from corpus.sources.luatvietnam import LuatVietnam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", default="data/processed/missing_laws.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    missing = set(json.load(open(a.missing)))
    refs = [r for r in Citations().table().to_dicts() if law_key(r["ref"]) in missing]
    print(f"{len(missing)} missing law keys -> {len(refs)} citation refs")
    lv = LuatVietnam()
    urls = lv.urls()
    with_url = [r for r in refs if lv.match(r, urls)]
    print(f"{len(with_url)} refs have a candidate page")
    if not a.dry_run:
        lv.build(with_url)


if __name__ == "__main__":
    main()
