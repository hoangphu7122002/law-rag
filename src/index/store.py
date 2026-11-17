"""The index cache of one (chunking, embedder) choice: data/index/<key>/{docstore.json, faiss/}.

Nodes are needed by every retriever (BM25 reads them); the vector index is built only for dense / hybrid.
"""
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from config import Config
from corpus import CorpusReader
from index.chunker import Chunker

ROOT = Path("data/index")


class IndexStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = ROOT / f"{cfg.chunk.kind}-{cfg.chunk.size}__{cfg.index_key()}"
        self._nodes: list[BaseNode] | None = None
        self._vector: VectorStoreIndex | None = None

    # ---- nodes ----
    def nodes(self) -> list[BaseNode]:
        if self._nodes is None:
            self._nodes = self._load_nodes() if self._docstore_path().exists() else self._build_nodes()
        return self._nodes

    def _docstore_path(self) -> Path:
        return self.dir / "docstore.json"

    def _build_nodes(self) -> list[BaseNode]:
        c = self.cfg.chunk
        docs = CorpusReader(lowercase=c.lowercase, flatten=c.flatten).read(self.cfg.corpus_limit)
        nodes = Chunker(c).split(docs)
        store = SimpleDocumentStore()
        store.add_documents(nodes)
        self.dir.mkdir(parents=True, exist_ok=True)
        store.persist(str(self._docstore_path()))
        print(f"index: {len(docs)} documents -> {len(nodes)} nodes, saved to {self.dir}")
        return nodes

    def _load_nodes(self) -> list[BaseNode]:
        store = SimpleDocumentStore.from_persist_path(str(self._docstore_path()))
        return sorted(store.docs.values(), key=lambda n: n.node_id)

    # ---- dense vectors ----
    def embedder(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        e = self.cfg.embed
        return HuggingFaceEmbedding(model_name=e.model, device=self.cfg.backend.device, max_length=e.max_len,
                                    normalize=e.normalize, embed_batch_size=e.batch,
                                    query_instruction=e.query_prefix or None, text_instruction=e.passage_prefix or None)

    def vector_index(self) -> VectorStoreIndex:
        if self._vector is None:
            faiss_dir = self.dir / "faiss"
            self._vector = self._load_vector(faiss_dir) if faiss_dir.exists() else self._build_vector(faiss_dir)
        return self._vector

    def _build_vector(self, faiss_dir: Path) -> VectorStoreIndex:
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore

        embed = self.embedder()
        dim = len(embed.get_text_embedding("Điều 1"))
        flat = faiss.IndexFlatL2(dim) if self.cfg.retrieve.metric == "l2" else faiss.IndexFlatIP(dim)
        storage = StorageContext.from_defaults(vector_store=FaissVectorStore(faiss_index=flat))
        index = VectorStoreIndex(self.nodes(), storage_context=storage, embed_model=embed,
                                 insert_batch_size=4096, show_progress=True)
        storage.persist(persist_dir=str(faiss_dir))
        return index

    def _load_vector(self, faiss_dir: Path) -> VectorStoreIndex:
        from llama_index.core import load_index_from_storage
        from llama_index.vector_stores.faiss import FaissVectorStore

        store = FaissVectorStore.from_persist_dir(str(faiss_dir))
        storage = StorageContext.from_defaults(vector_store=store, persist_dir=str(faiss_dir))
        return load_index_from_storage(storage, embed_model=self.embedder())
