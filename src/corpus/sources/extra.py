"""Extra public Vietnamese legal datasets (training data only, never the RAG corpus) -> data/raw/extra/<name>/.

Downloaded as-is from HuggingFace. Before any of these feed training, they must pass the test decontamination
filter (rule in AGENTS.md): they may overlap the RES QA, which itself comes from public legal Q&A sites.

    ExtraData().download()          # all
    ExtraData().download(["zalo"])  # one
"""
from pathlib import Path

RAW = Path("data/raw/extra")

DATASETS = {  # name -> (HF repo, use, license as stated on the card)
    "zalo": ("minhnguyent546/zalo-ai-legal-text-retrieval-2021", "retrieval pairs: 3,196 questions -> article, 61k articles", "none stated"),
    "bkai": ("tmnam20/BKAI-Legal-Retrieval", "retrieval pairs from thuvienphapluat Q&A (BKAI)", "none stated"),
    "legal_qa": ("thangvip/vietnamese-legal-qa", "LLM-generated QA per article of 2024+ laws (9.7k articles)", "none stated"),
}


class ExtraData:
    def download(self, names: list[str] | None = None) -> None:
        from huggingface_hub import snapshot_download

        for name in names or list(DATASETS):
            repo, use, lic = DATASETS[name]
            out = RAW / name
            print(f"{name}: {repo} ({use}; license {lic}) -> {out}")
            snapshot_download(repo, repo_type="dataset", local_dir=out)
