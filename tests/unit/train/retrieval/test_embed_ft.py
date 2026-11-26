import json

import pytest

from train.retrieval.embed_ft import VARIANTS, load_groups, variant_args


def write(tmp_path, rows):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def row(i, nneg):
    return {"query": f"q{i}", "pos": [f"p{i}"], "neg": [f"n{i}_{j}" for j in range(nneg)]}


def test_groups_become_anchor_positive_negative_columns(tmp_path):
    ex = load_groups(write(tmp_path, [row(0, 7)]), negatives=2)
    assert ex == [{"anchor": "q0", "positive": "p0", "negative_1": "n0_0", "negative_2": "n0_1"}]


def test_zero_negatives_gives_pairs_only(tmp_path):
    ex = load_groups(write(tmp_path, [row(0, 7)]), negatives=0)
    assert ex == [{"anchor": "q0", "positive": "p0"}]


def test_groups_with_too_few_negatives_are_dropped_so_columns_stay_rectangular(tmp_path):
    ex = load_groups(write(tmp_path, [row(0, 7), row(1, 1), row(2, 0)]), negatives=3)
    assert [e["anchor"] for e in ex] == ["q0"]


def test_limit_and_seeded_shuffle_are_reproducible(tmp_path):
    p = write(tmp_path, [row(i, 3) for i in range(20)])
    a, b = load_groups(p, 1, limit=5, seed=1), load_groups(p, 1, limit=5, seed=1)
    assert a == b and len(a) == 5 and a != load_groups(p, 1, limit=5, seed=2)


def test_variants_cover_lora_full_and_in_batch_only():
    assert {"lora", "full"} <= {v["mode"] for v in VARIANTS.values()}
    assert any(v["negatives"] == 0 for v in VARIANTS.values())   # control: in-batch negatives only
    assert all(v["lr"] > 0 and v["epochs"] > 0 for v in VARIANTS.values())


def test_variant_args_rejects_unknown_name():
    with pytest.raises(KeyError):
        variant_args("nope")
    assert variant_args("lora32")["r"] == 32


def test_interpolation_moves_from_base_to_tuned_weights():
    import numpy as np
    from train.retrieval.embed_ft import interpolate

    base, tuned = {"w": np.array([0.0, 4.0]), "step": 3}, {"w": np.array([2.0, 0.0]), "step": 9}
    assert interpolate(base, tuned, 0.0)["w"].tolist() == [0.0, 4.0]
    assert interpolate(base, tuned, 1.0)["w"].tolist() == [2.0, 0.0]
    assert interpolate(base, tuned, 0.25)["w"].tolist() == [0.5, 3.0]
    assert interpolate(base, tuned, 0.5)["step"] == 9          # non-float entries (counters) come from the tuned side


def test_interpolation_rejects_mismatched_weights():
    import numpy as np
    from train.retrieval.embed_ft import interpolate

    with pytest.raises(ValueError):
        interpolate({"a": np.zeros(1)}, {"b": np.zeros(1)}, 0.5)


def test_conservative_variants_exist_query_only_and_interpolated():
    assert VARIANTS["query32"]["mode"] == "query"                  # document tower frozen = the index stays valid
    assert VARIANTS["lora32_low"]["lr"] < VARIANTS["lora32"]["lr"]
    assert all(0 < a < 1 for v in VARIANTS.values() for a in v.get("alphas", []))
    assert VARIANTS["lora32"].get("alphas")
