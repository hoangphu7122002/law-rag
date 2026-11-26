"""README architecture figure (AWS-diagram style).  python docs/assets/make_figures.py   (needs Google Chrome)
Facts: docker/compose.yml (three vLLM endpoints + API), configs/exp/A5_v3b.yaml (dense, candidates 50, k 5, 2 shots),
src/serve/api.py (POST /ask, MIN_RERANK_SCORE 0.001), README / report (corpus, SFT recipe v3b, two DPO rounds)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from awsfig import BLUE, GREY, INK, ORANGE, PURPLE, RED, TEAL, Fig  # noqa: E402

c = Fig(1640, 1190, HERE / "icons")
c.title("law-rag: one question, and how the model behind it was built",
        "serving: one L40S running docker/compose.yml · offline: data and training scripts in this repo")

# ---------------- serving band
c.group(20, 110, 180, 610, "Client", stroke=GREY, dash="6,4")
c.group(210, 110, 1620, 610, "GPU box: one L40S, docker/compose.yml", stroke=INK, icon="onprem/container/docker")
c.group(720, 150, 1010, 590, "vLLM endpoints (OpenAI API)", stroke=TEAL, dash="7,5")

c.node(100, 320, "onprem/client/user", "Client", ["curl, app"])
c.node(440, 320, "programming/framework/fastapi", "QA service", ["POST /ask", "src/serve/api.py"])
c.node(440, 530, "generic/storage/storage", "FAISS index", ["one passage per Điều"], size=48)
c.node(865, 220, "aws/ml/sagemaker-model", "Embedder", ["Vietnamese_Embedding"], size=50)
c.node(865, 340, "aws/ml/sagemaker-model", "Reranker", ["Vietnamese_Reranker"], size=50)
c.node(865, 480, "aws/ml/sagemaker-model", "Generator", ["Qwen3.5-2B, SFT + 2 DPO"], size=50)

c.arrow([(130, 305), (405, 305)])
c.plate(265, 296, "question")
c.arrow([(405, 335), (130, 335)], color=GREY, dash="7,5")
c.plate(265, 360, "answer + sources")
c.step(230, 270, 1)

# trunk from the service to the three endpoints
c.arrow([(475, 305), (560, 305), (560, 220), (835, 220)])
c.arrow([(560, 305), (560, 340), (835, 340)])
c.arrow([(475, 335), (540, 335), (540, 480), (835, 480)])
c.arrow([(440, 410), (440, 500)])
c.plate(640, 212, "embed")
c.step(640, 185, 2)
c.plate(480, 460, "top 50", 14)
c.step(405, 460, 3)
c.plate(640, 332, "rerank, keep 5")
c.step(640, 305, 4)
c.plate(637, 472, "5 passages + 2 shots", 14)
c.step(640, 445, 5)

# refusal note
c.arrow([(895, 340), (1080, 340)], color=RED, dash="4,4")
c.text(1095, 335, "best score < 0.001: refuse", 15, "bold", "start", RED)
c.text(1095, 355, "(\"the context does not cover this\")", 14, "normal", "start", "#546e7a")

# answer shape
c.rect(1080, 450, 1600, 560, stroke="#cfd8dc", fill="#f7f9fa", rx=6)
c.text(1095, 477, "Answer", 15, "bold", "start", "#546e7a")
c.text(1095, 502, "Căn cứ tại khoản k Điều n <văn bản> quy định như sau:", 15, "normal", "start", INK, italic=True)
c.text(1095, 524, "<quoted provision> ... Như vậy, ...", 15, "normal", "start", INK, italic=True)
c.text(1095, 546, "+ sources (law, Điều, rerank score) and timings", 14, "normal", "start", "#546e7a")

# ---------------- offline band
c.group(210, 650, 1620, 1010, "Offline, before serving: data and training (scripts/)", stroke="#78909c", dash="6,4")

c.node(320, 740, "generic/storage/storage", "Laws corpus", ["course 16,441 texts", "+ vbpl + web crawl"], size=48)
c.node(560, 740, "programming/language/python", "Chunk + embed", ["clean, one passage", "per Điều, de-dup"], size=48)
c.arrow([(350, 740), (525, 740)])
c.arrow([(595, 740), (680, 740), (680, 630), (440, 630), (440, 598)], color=PURPLE, dash="8,5")
c.plate(560, 625, "build index", 14)
c.step(700, 700, 6)

c.node(320, 900, "generic/storage/storage", "Train split", ["decontaminated", "vs test (law, Điều)"], size=48)
c.node(560, 900, "aws/ml/sagemaker-training-job", "LoRA SFT", ["k 3-10, gold 70%,", "0-3 shots, refusals"], size=50)
c.node(800, 900, "programming/language/python", "Sample 8", ["answers per prompt"], size=48)
c.node(1020, 900, "programming/language/python", "Pairs", ["best vs worst", "by ROUGE-L"], size=48)
c.node(1240, 900, "aws/ml/sagemaker-training-job", "DPO", ["beta .1, lr 2e-5"], size=50)
c.node(1470, 900, "aws/ml/sagemaker-model", "dpo_a2 weights", ["final model"], size=50)
for x0, x1 in ((350, 525), (595, 765), (835, 985), (1055, 1205), (1275, 1435)):
    c.arrow([(x0, 900), (x1, 900)])
c.arrow([(1240, 966), (1240, 990), (800, 990), (800, 968)], color=GREY, dash="7,5")
c.plate(1050, 995, "round 2: sample again from the round-1 model", 14)
c.arrow([(1470, 865), (1470, 630), (1050, 630), (1050, 480), (895, 480)], color=PURPLE, dash="8,5")
c.plate(1260, 625, "served weights", 14)
c.step(1495, 780, 7)

# ---------------- legend
c.legend_steps(40, 1060, [
    "The client sends a question to POST /ask",
    "The service embeds it on the embedder endpoint",
    "Dense search returns the top 50 passages",
    "The reranker keeps the best 5 (or the service refuses)",
    "The generator answers from 5 passages and 2 example Q/A pairs",
    "Offline: the corpus is chunked per Điều and indexed",
    "Offline: SFT, then two on-policy DPO rounds, give the weights",
], cols=2, colw=560)
c.legend_arrows(1240, 1045, [("request or call", INK, ""), ("reply or loop", GREY, "7,5"),
                            ("artifact deployed", PURPLE, "8,5"), ("refusal", RED, "4,4")])
c.save(HERE / "architecture")
