from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from retrieve.vllm_rerank import VllmRerank


def test_reorders_by_server_scores(monkeypatch):
    class Res:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}

    sent = {}
    monkeypatch.setattr("httpx.post", lambda url, json, timeout: sent.update(url=url, body=json) or Res())
    nodes = [NodeWithScore(node=TextNode(text="a", id_="a"), score=0.5),
             NodeWithScore(node=TextNode(text="b", id_="b"), score=0.4)]
    out = VllmRerank(url="http://x", model="m", top_n=2).postprocess_nodes(nodes, QueryBundle("q"))
    assert [n.node.node_id for n in out] == ["b", "a"] and out[0].score == 0.9
    assert sent["url"] == "http://x/rerank" and sent["body"]["documents"] == ["a", "b"]
