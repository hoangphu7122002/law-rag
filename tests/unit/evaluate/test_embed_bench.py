import numpy as np

from evaluate.embed_bench import EmbedBench
from records import Passage


def P(pid, doc, art, extra=""):
    return Passage(pid, doc, f"Điều {art}. Tiêu đề\n{extra}", 0.0)


class Topic:
    """Fake embedder: a text is a 1-hot vector on the first known keyword it contains (the only thing we fake)."""

    def __init__(self, words):
        self.words, self.calls = words, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), len(self.words)))
        for i, t in enumerate(texts):
            for j, w in enumerate(self.words):
                if w in t:
                    out[i, j] = 1.0
                    break
        return out


POOL = [P("a", "10_2020_NĐ", 1, "thuế"), P("b", "10_2020_NĐ", 2, "phí"), P("c", "20_2021_TT", 3, "đất"), P("d", "20_2021_TT", 4, "nước")]


def test_a_perfect_embedder_puts_the_gold_dieu_first():
    bench = EmbedBench(POOL, queries={1: "thuế là gì", 2: "đất đai"}, gold={1: {"10/2020/NĐ#1"}, 2: {"20/2021/TT#3"}})
    m = bench.run(Topic(["thuế", "phí", "đất", "nước"]))
    assert (m["art@1"], m["art@5"], m["mrr"]) == (1.0, 1.0, 1.0)


def test_a_wrong_embedder_scores_by_the_rank_of_the_gold_dieu():
    bench = EmbedBench(POOL, queries={1: "phí"}, gold={1: {"10/2020/NĐ#1"}})      # query matches Điều 2, gold is Điều 1 (ties after it)
    m = bench.run(Topic(["thuế", "phí", "đất", "nước"]))
    assert m["art@1"] == 0.0 and 0.0 < m["mrr"] < 1.0 and m["art@5"] == 1.0


def test_prefixes_go_to_the_right_side_only():
    emb = Topic(["thuế", "phí"])
    EmbedBench(POOL, queries={1: "thuế là gì"}, gold={1: {"10/2020/NĐ#1"}}).run(emb, query_prefix="Instruct: tìm điều luật\nQuery: ", doc_prefix="passage: ")
    docs, queries = emb.calls[0], emb.calls[1]
    assert all(t.startswith("passage: ") for t in docs) and queries == ["Instruct: tìm điều luật\nQuery: thuế là gì"]


def test_texts_are_embedded_in_batches_of_the_requested_size():
    emb = Topic(["thuế"])
    EmbedBench(POOL, queries={1: "a", 2: "b", 3: "c"}, gold={1: {"x#1"}, 2: {"x#1"}, 3: {"x#1"}}).run(emb, batch=3)
    assert [len(c) for c in emb.calls][:3] == [3, 1, 3]       # 4 passages in 3 + 1, then the 3 queries (single-query latency probes follow)


class Clocked(Topic):
    """Each embedding call costs `per_text` seconds per text plus `fixed` seconds on a fake clock."""

    def __init__(self, words, clock, per_text=0.01, fixed=0.5):
        super().__init__(words)
        self.clock, self.per_text, self.fixed = clock, per_text, fixed

    def __call__(self, texts):
        self.clock[0] += self.fixed + self.per_text * len(texts)
        return super().__call__(texts)


def test_latency_is_reported_as_passage_throughput_and_single_query_milliseconds():
    now = [0.0]
    emb = Clocked(["thuế", "phí"], now)
    bench = EmbedBench(POOL, queries={1: "thuế", 2: "phí"}, gold={1: {"10/2020/NĐ#1"}, 2: {"10/2020/NĐ#2"}}, clock=lambda: now[0])
    m = bench.run(emb, batch=64)
    assert round(m["passages_per_s"], 2) == round(4 / (0.5 + 0.04), 2)       # 4 passages in one batch
    assert round(m["query_ms_p50"]) == 510                                     # one query alone: 0.5 + 0.01 s


def test_pool_is_candidates_plus_gold_articles_without_duplicates():
    from evaluate.embed_bench import build_pool

    cand = {1: [POOL[0], POOL[1]], 2: [POOL[1], POOL[2]]}
    articles = {"20/2021/TT#4": [POOL[3]], "10/2020/NĐ#1": [POOL[0]]}          # gold article lookup (corpus side)
    gold = {1: {"10/2020/NĐ#1"}, 2: {"20/2021/TT#4"}}
    pool = build_pool(cand, gold, articles)
    assert [p.pid for p in pool] == ["a", "b", "c", "d"]                       # candidate order, then missing gold; "a", "b" once


def test_a_gold_article_missing_from_the_corpus_is_skipped_not_fatal():
    from evaluate.embed_bench import build_pool

    pool = build_pool({1: [POOL[0]]}, {1: {"99/2000/XX#7"}}, {})
    assert [p.pid for p in pool] == ["a"]


class FakeEmbClient:
    """openai-style client: embeddings.create(model=, input=[...]) -> .data[i].embedding (the HTTP boundary we fake)."""

    def __init__(self):
        self.calls = []
        self.embeddings = type("E", (), {"create": self._create})()

    def _create(self, model, input, **kw):
        self.calls.append((model, list(input), kw))
        return type("R", (), {"data": [type("D", (), {"embedding": [float(len(t)), 1.0]})() for t in input]})()


def test_server_embedder_sends_texts_to_the_model_and_returns_an_array():
    from evaluate.embed_bench import ServerEmbedder

    c = FakeEmbClient()
    out = ServerEmbedder(c, "Qwen/Qwen3-Embedding-4B")(["ab", "abcd"])
    assert out.shape == (2, 2) and out[1, 0] == 4.0
    assert c.calls[0][0] == "Qwen/Qwen3-Embedding-4B"


def test_server_embedder_can_truncate_to_a_matryoshka_dimension_and_cap_input_tokens():
    from evaluate.embed_bench import ServerEmbedder

    c = FakeEmbClient()
    out = ServerEmbedder(c, "m", dims=1, max_len=512)(["ab"])
    assert out.shape == (1, 1)
    assert c.calls[0][2]["extra_body"] == {"truncate_prompt_tokens": 512}


def test_model_specs_load_from_yaml_with_defaults(tmp_path):
    from evaluate.embed_bench import load_specs

    f = tmp_path / "m.yaml"
    f.write_text("- {name: e5, model: intfloat/multilingual-e5-large-instruct, kind: st, query_prefix: 'query: ', doc_prefix: 'passage: '}\n"
                 "- {name: q4b, model: Qwen/Qwen3-Embedding-4B, kind: server}\n")
    specs = load_specs(f)
    assert [s["name"] for s in specs] == ["e5", "q4b"]
    assert specs[0]["query_prefix"] == "query: " and specs[1]["query_prefix"] == "" and specs[1]["batch"] == 32


def test_hf_cache_dir_of_a_model_lives_under_the_given_home(tmp_path):
    from evaluate.embed_bench import hf_cache_dir

    assert hf_cache_dir("Qwen/Qwen3-Embedding-4B", tmp_path) == tmp_path / "hub" / "models--Qwen--Qwen3-Embedding-4B"


def test_purge_removes_only_that_models_cache_and_only_under_our_home(tmp_path):
    from evaluate.embed_bench import purge_model

    mine, other = tmp_path / "hub" / "models--Qwen--Qwen3-Embedding-4B", tmp_path / "hub" / "models--BAAI--bge-m3"
    for d in (mine, other):
        (d / "blobs").mkdir(parents=True)
        (d / "blobs" / "w.bin").write_text("x")
    assert purge_model("Qwen/Qwen3-Embedding-4B", tmp_path) is True
    assert not mine.exists() and other.exists()
    assert purge_model("Qwen/Qwen3-Embedding-4B", tmp_path) is False        # already gone


def test_a_pool_survives_a_save_and_load_round_trip(tmp_path):
    from evaluate.embed_bench import load_pool, save_pool

    queries, gold = {1: "thuế là gì", 7: "đất đai"}, {1: {"10/2020/NĐ#1"}, 7: {"20/2021/TT#3", "20/2021/TT#4"}}
    save_pool(tmp_path / "pool.json", POOL, queries, gold)
    p2, q2, g2, k2, _ = load_pool(tmp_path / "pool.json")
    assert [(p.pid, p.doc, p.text) for p in p2] == [(p.pid, p.doc, p.text) for p in POOL]
    assert q2 == queries and g2 == gold                      # int qids and set-valued gold restored (JSON turns keys into strings)
    assert k2 == ["10/2020/NĐ#1", "10/2020/NĐ#2", "20/2021/TT#3", "20/2021/TT#4"]      # article keys computed once, at export


def test_a_bench_with_precomputed_keys_never_needs_the_corpus_helpers():
    keys = ["x#1", "x#2"]
    pool = [Passage("a", "whatever", "no heading here thuế", 0.0), Passage("b", "whatever", "no heading here phí", 0.0)]
    m = EmbedBench(pool, {1: "thuế"}, {1: {"x#1"}}, keys=keys).run(Topic(["thuế", "phí"]))
    assert m["art@1"] == 1.0


def test_a_failing_model_is_recorded_and_the_others_still_run():
    from evaluate.embed_bench import run_specs

    bench = EmbedBench(POOL, {1: "thuế"}, {1: {"10/2020/NĐ#1"}})

    def make(spec):
        if spec["model"] == "broken":
            raise RuntimeError("trust_remote_code failed")
        return Topic(["thuế", "phí"])

    specs = [{"name": "ok", "model": "ok", "query_prefix": "", "doc_prefix": "", "batch": 8},
             {"name": "bad", "model": "broken", "query_prefix": "", "doc_prefix": "", "batch": 8},
             {"name": "ok2", "model": "ok2", "query_prefix": "", "doc_prefix": "", "batch": 8}]
    rows = run_specs(bench, specs, make)
    assert [r["name"] for r in rows] == ["ok", "bad", "ok2"]
    assert rows[0]["art@1"] == 1.0 and "trust_remote_code" in rows[1]["error"] and rows[2]["art@1"] == 1.0


def test_embedders_are_closed_after_use_even_when_they_fail():
    from evaluate.embed_bench import run_specs

    closed = []

    class Closing(Topic):
        def close(self):
            closed.append(True)

    class Boom(Closing):
        def __call__(self, texts):
            raise RuntimeError("cuda oom")

    bench = EmbedBench(POOL, {1: "thuế"}, {1: {"10/2020/NĐ#1"}})
    spec = {"name": "x", "model": "x", "query_prefix": "", "doc_prefix": "", "batch": 8}
    run_specs(bench, [spec, spec], lambda s: Boom(["a"]) if len(closed) == 0 else Closing(["thuế"]))
    assert closed == [True, True]


def test_metrics_are_also_reported_per_question_group():
    bench = EmbedBench(POOL, queries={1: "thuế", 2: "phí", 3: "đất"}, gold={1: {"10/2020/NĐ#1"}, 2: {"10/2020/NĐ#1"}, 3: {"20/2021/TT#3"}},
                       groups={"first": [1, 3], "second": [2]})
    m = bench.run(Topic(["thuế", "phí", "đất", "nước"]))
    assert m["groups"]["first"]["art@1"] == 1.0 and m["groups"]["second"]["art@1"] == 0.0       # question 2 asks "phí", gold is the "thuế" article
    assert m["groups"]["first"]["n"] == 2 and m["groups"]["second"]["n"] == 1 and m["n"] == 3


def test_groups_survive_a_pool_round_trip(tmp_path):
    from evaluate.embed_bench import load_pool, save_pool

    save_pool(tmp_path / "p.json", POOL, {1: "a"}, {1: {"10/2020/NĐ#1"}}, groups={"dev_clean": [1]})
    assert load_pool(tmp_path / "p.json")[4] == {"dev_clean": [1]}


def test_queries_can_use_a_different_embedder_than_documents_for_query_only_tuning():
    docs_emb, query_emb = Topic(["thuế", "phí", "đất", "nước"]), Topic(["thuế", "phí", "đất", "nước"])
    bench = EmbedBench(POOL, queries={1: "thuế là gì"}, gold={1: {"10/2020/NĐ#1"}})
    m = bench.run(docs_emb, query_embedder=query_emb)
    assert m["art@1"] == 1.0
    assert all("thuế là gì" not in t for call in docs_emb.calls for t in call)   # documents only went to the document tower
    assert query_emb.calls and all(len(call) == 1 or call[0] == "thuế là gì" for call in query_emb.calls)


def test_a_run_reports_the_rank_of_the_gold_dieu_per_question():
    bench = EmbedBench(POOL, queries={1: "thuế là gì", 2: "phí"}, gold={1: {"10/2020/NĐ#1"}, 2: {"20/2021/TT#3"}})
    ranks = bench.run(Topic(["thuế", "phí", "đất", "nước"]))["ranks"]
    assert ranks[1] == 1 and ranks[2] > 1


def test_paired_bootstrap_detects_a_consistent_gain_and_not_noise():
    from evaluate.embed_bench import paired_bootstrap

    base = {i: 3 for i in range(100)}
    better = {i: 1 for i in range(100)}                       # always rank 1 vs always rank 3
    d = paired_bootstrap(better, base, "art@1")
    assert d["diff"] == 1.0 and d["lo"] > 0 and d["p"] < 0.01
    same = paired_bootstrap(base, dict(base), "mrr")
    assert same["diff"] == 0.0 and same["lo"] <= 0 <= same["hi"]
    flip = {i: (1 if i % 2 else 3) for i in range(100)}
    flip2 = {i: (3 if i % 2 else 1) for i in range(100)}       # opposite half each: mean difference 0, wide interval
    n = paired_bootstrap(flip, flip2, "art@1")
    assert n["lo"] < 0 < n["hi"]


def test_paired_bootstrap_only_compares_questions_both_runs_have():
    from evaluate.embed_bench import paired_bootstrap

    d = paired_bootstrap({1: 1, 2: 1, 9: 1}, {1: 2, 2: 2, 5: 1}, "art@1")
    assert d["n"] == 2
