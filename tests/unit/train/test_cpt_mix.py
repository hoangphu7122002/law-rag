from train.cpt_mix import mix, take


def test_take_stops_at_the_token_budget_and_is_deterministic():
    docs = [[1] * 10 for _ in range(20)]
    a, b = take(docs, 55, seed=1), take(docs, 55, seed=1)
    assert a == b and 55 <= sum(map(len, a)) < 55 + 10


def test_take_returns_everything_when_the_budget_exceeds_the_data():
    docs = [[1] * 10 for _ in range(3)]
    assert sum(map(len, take(docs, 10**6, seed=0))) == 30


def test_mix_packs_all_sources_into_fixed_length_sequences_and_reports_tokens_per_source():
    src = {"raw": [[1] * 30], "aug": [[2] * 30, [2] * 30], "replay": [[3] * 30]}
    seqs, stats = mix(src, {"raw": 30, "aug": 60, "replay": 30}, seq_len=20, eos=0, seed=0)
    assert all(len(s) == 20 for s in seqs)
    assert stats == {"raw": 30, "aug": 60, "replay": 30}
    assert {t for s in seqs for t in s} == {0, 1, 2, 3}


def test_mix_shuffles_documents_so_sources_are_interleaved():
    src = {"a": [[1] * 10 for _ in range(30)], "b": [[2] * 10 for _ in range(30)]}
    seqs, _ = mix(src, {"a": 300, "b": 300}, seq_len=10, eos=0, seed=3)
    firsts = [s[0] for s in seqs]
    assert firsts != sorted(firsts)


def test_mix_np_packs_numpy_documents_into_a_uint32_matrix_with_eos_and_reports_tokens():
    import numpy as np

    from train.cpt_mix import mix_np
    src = {"a": [np.full(30, 1, dtype=np.uint32)], "b": [np.full(30, 2, dtype=np.uint32), np.full(30, 2, dtype=np.uint32)]}
    arr, stats = mix_np(src, {"a": 30, "b": 60}, seq_len=20, eos=0, seed=0)
    assert arr.dtype == np.uint32 and arr.shape[1] == 20 and stats == {"a": 30, "b": 60}
    assert set(np.unique(arr)) == {0, 1, 2}


def test_npy_dataset_returns_labels_equal_to_input_ids(tmp_path):
    import numpy as np

    from train.cpt_mix import NpyDataset
    np.save(tmp_path / "m.npy", np.arange(40, dtype=np.uint32).reshape(2, 20))
    ds = NpyDataset(str(tmp_path / "m.npy"), start=1, stop=2)
    assert len(ds) == 1 and ds[0]["input_ids"].tolist() == list(range(20, 40)) and ds[0]["labels"].tolist() == list(range(20, 40))
