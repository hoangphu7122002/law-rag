"""Config -> LlamaIndex retriever + reranker. The only place that knows which class each `kind` means."""
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever

from config import Config
from evaluate.metrics import normalize_text
from index import IndexStore


class RetrieverFactory:
    def __init__(self, cfg: Config, store: IndexStore):
        self.cfg, self.store = cfg, store
        r = cfg.retrieve
        # fetch more than k when something re-orders the list afterwards
        self.top_n = r.candidates if (r.kind == "hybrid" or cfg.rerank.model) else r.k

    def retriever(self) -> BaseRetriever:
        kind = self.cfg.retrieve.kind
        if kind == "dense":
            return self.dense(self.top_n)
        if kind == "bm25":
            return self.bm25(self.top_n)
        if kind == "hybrid":
            n = self.cfg.retrieve.candidates
            return QueryFusionRetriever([self.dense(n), self.bm25(n)], mode="reciprocal_rerank",
                                        similarity_top_k=self.top_n, num_queries=1, use_async=False)
        raise ValueError(f"unknown retrieve.kind {kind!r}")

    def reranker(self) -> BaseNodePostprocessor | None:
        r = self.cfg.rerank
        if not r.model:
            return None
        if r.engine == "vllm":
            from retrieve.vllm_rerank import VllmRerank

            return VllmRerank(url=self.cfg.backend.rerank_url, model=r.model, top_n=self.cfg.retrieve.k, max_len=r.max_len)
        from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

        return SentenceTransformerRerank(model=r.model, top_n=self.cfg.retrieve.k, device=self.cfg.backend.device,
                                         keep_retrieval_score=True)

    def dense(self, k: int) -> BaseRetriever:
        return self.store.vector_index().as_retriever(similarity_top_k=k)

    def bm25(self, k: int) -> BaseRetriever:
        from llama_index.retrievers.bm25 import BM25Retriever

        # Vietnamese: no stemming, our normalisation (lower-case, no punctuation), syllables as tokens
        return BM25Retriever.from_defaults(nodes=self.store.nodes(), similarity_top_k=k, skip_stemming=True,
                                           tokenizer=lambda text: normalize_text(text).split())
