"""Demo: send example questions to the running QA API and write a readable Markdown transcript.

    PYTHONPATH=src python scripts/serve/demo.py docs/report/demo_questions.json --url http://localhost:8080 > docs/report/demo.md
questions file: JSON list of {"qid": <dev question id>} (text + reference taken from the dev split) or {"question": "<free text>"}.
Dev only, never the test split.
"""
import argparse
import json
import urllib.request


def render(n: int, question: str, reply: dict, reference: str | None = None) -> str:
    t = reply["timings"]
    lines = [f"### Q{n}. {question}", "", "**Answer**", "", reply["answer"].strip(), "",
             "**Sources** (reranked): " + "; ".join(f"{s['doc']} ({s['score']:.3f})" for s in reply["sources"]), "",
             f"**Latency**: {t['total_s']:.1f} s (retrieve {t['retrieve_s']:.2f} · rerank {t['rerank_s']:.2f} · generate {t['generate_s']:.2f})"]
    if reference is not None:
        from evaluate.metrics import rouge_l_score

        lines += ["", f"ROUGE-L vs reference: {rouge_l_score(reply['answer'], reference):.3f}", "",
                  "<details><summary>Reference answer</summary>", "", reference.strip(), "", "</details>"]
    return "\n".join(lines) + "\n"


def ask(url: str, question: str) -> dict:
    req = urllib.request.Request(f"{url}/ask", json.dumps({"question": question}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--url", default="http://localhost:8080")
    a = ap.parse_args()
    items = json.load(open(a.questions, encoding="utf-8"))
    dev = None
    print("# Demo transcript\n")
    for n, it in enumerate(items, 1):
        ref = None
        if "qid" in it:
            from corpus import QASet

            dev = dev or {q.qid: q for q in QASet().load("dev")}
            q = dev[it["qid"]]
            text, ref = q.text, q.answer
        else:
            text = it["question"]
        print(render(n, text, ask(a.url, text), ref))


if __name__ == "__main__":
    main()
