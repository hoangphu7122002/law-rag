from ops.box import CODE_EXCLUDES


def test_only_the_top_level_data_dir_is_excluded_from_the_code_push():
    """An unanchored `data` would also skip scripts/data/ (a code folder since the 2026-11-26 reorganisation)."""
    assert "/data" in CODE_EXCLUDES and "data" not in CODE_EXCLUDES
