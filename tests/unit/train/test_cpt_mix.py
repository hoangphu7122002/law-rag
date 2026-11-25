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
