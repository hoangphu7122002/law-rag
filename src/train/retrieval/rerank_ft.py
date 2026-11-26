"""Fine-tune the cross-encoder reranker on src/train/rerank_data.py groups (sentence-transformers CrossEncoderTrainer).

    python src/train/rerank_ft.py --data data/train/rr_v0 --base AITeamVN/Vietnamese_Reranker --out data/models/rr_v0

Loss: binary cross-entropy on (query, passage) pairs, 1 positive : N hard negatives (pos_weight = N).
Val: CrossEncoderRerankingEvaluator on decontaminated dev (rerank the real dense candidates; MRR@10 / NDCG@10);
the best-val checkpoint is kept. Eval in the pipeline: `cli run <exp> --retrieval-only --set rerank.engine=local
--set rerank.model=data/models/rr_v0` (compare with the base model on the same engine).
"""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--eval-steps", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N groups / val queries")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    t0, data, out = time.time(), Path(a.data), Path(a.out)
    read = lambda f: [json.loads(x) for x in (data / f).read_text(encoding="utf-8").splitlines()]
    groups, val = read("train.jsonl"), read("val.jsonl")
    if a.limit:
        groups, val = groups[: a.limit], val[: max(4, a.limit // 4)]
    pairs = {"query": [], "passage": [], "label": []}
    for g in groups:
        for text, y in [(p, 1.0) for p in g["pos"]] + [(n, 0.0) for n in g["neg"]]:
            pairs["query"].append(g["query"]), pairs["passage"].append(text), pairs["label"].append(y)
    ds = Dataset.from_dict(pairs)
    neg_per_pos = max(1, round(sum(len(g["neg"]) for g in groups) / max(1, sum(len(g["pos"]) for g in groups))))

    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_length)
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(neg_per_pos)))
    ev = CrossEncoderRerankingEvaluator(samples=[{"query": v["query"], "positive": v["pos"], "documents": v["docs"]}
                                                 for v in val], at_k=10, name="dev", always_rerank_positives=False)
    every = 5 if a.limit else a.eval_steps
    args = CrossEncoderTrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, max_steps=10 if a.limit else -1,
        per_device_train_batch_size=a.batch, learning_rate=a.lr, warmup_ratio=0.1, lr_scheduler_type="linear",
        bf16=torch.cuda.is_available(), eval_strategy="steps", eval_steps=every, save_strategy="steps", save_steps=every,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_dev_ndcg@10",
        logging_steps=max(1, every // 3), report_to=[], seed=a.seed, dataloader_drop_last=False)
    print(f"base before training: {ev(model)}")
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss, evaluator=ev)
    result = trainer.train()
    final = ev(model)
    model.save_pretrained(str(out))
    import shutil

    shutil.rmtree(out / "ckpt", ignore_errors=True)
    hist = trainer.state.log_history
    rec = {"base": a.base, "data": str(data), "groups": len(groups), "pairs": len(ds), "val": len(val),
           "neg_per_pos": neg_per_pos, "args": vars(a), "train_loss": result.training_loss, "steps": result.global_step,
           "best_checkpoint": trainer.state.best_model_checkpoint, "final_val": final,
           "eval": [{k: v for k, v in r.items() if k.startswith("eval_dev") or k == "step"} for r in hist if "eval_dev_ndcg@10" in r],
           "minutes": round((time.time() - t0) / 60, 1)}
    (out / "train_report.json").write_text(json.dumps(rec, indent=1))
    print(f"done: {out} val {final} in {rec['minutes']} min")


if __name__ == "__main__":
    main()
