"""Cross-encoder rerank through a vLLM pooling server (`/rerank`, Jina/Cohere format) instead of in-process."""
from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


class VllmRerank(BaseNodePostprocessor):
    url: str = Field(description="server root, e.g. http://localhost:8200")
    model: str
    top_n: int = 5
    max_len: int = 1024
    timeout: float = 120.0

    @classmethod
    def class_name(cls) -> str:
        return "VllmRerank"

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None):
        if not nodes or query_bundle is None:
            return nodes[: self.top_n]
        import httpx

        body = {"model": self.model, "query": query_bundle.query_str, "top_n": self.top_n,
                "documents": [n.node.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes], "truncate_prompt_tokens": self.max_len}
        res = httpx.post(f"{self.url}/rerank", json=body, timeout=self.timeout)
        res.raise_for_status()
        out = []
        for r in res.json()["results"]:
            node = nodes[r["index"]]
            node.node.metadata["retrieval_score"] = node.score
            out.append(NodeWithScore(node=node.node, score=r["relevance_score"]))
        return out
