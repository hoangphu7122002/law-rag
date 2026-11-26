"""Every repo path that a queue file, the Makefile or a README tells you to run must exist (structure tests after the 2026-11-26 reorganisation)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = re.compile(r"(?<![\w/.-])((?:scripts|configs|src)/[A-Za-z0-9_./-]+\.(?:py|sh|yaml|yml|txt))")
# paths that only exist on the remote box / are produced by a run, or are placeholders in prose
IGNORE = ("configs/queue_a2e.txt", "configs/exp/<", "scripts/<", "configs/queues/<", "configs/backends/*", "src/train/<")


def referenced(files):
    for f in files:
        for m in PATH.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            p = m.group(1)
            if not p.endswith(("/", ".")) and not any(p.startswith(i) for i in IGNORE):
                yield f, p


def test_queue_files_only_reference_existing_repo_paths():
    files = [f for f in (ROOT / "configs" / "queues").rglob("*.txt")]
    missing = sorted({(str(f.relative_to(ROOT)), p) for f, p in referenced(files) if not (ROOT / p).exists()})
    assert not missing, f"queue files reference missing paths: {missing[:10]}"


def test_makefile_agents_and_readme_reference_existing_repo_paths():
    files = [ROOT / "Makefile", ROOT / "AGENTS.md", ROOT / "README.md"]
    missing = sorted({(f.name, p) for f, p in referenced([f for f in files if f.exists()]) if not (ROOT / p).exists()})
    assert not missing, f"missing: {missing[:10]}"


def test_every_scripts_and_queue_folder_has_an_index():
    for d in [*(ROOT / "scripts").iterdir(), *(ROOT / "configs" / "queues").iterdir()]:
        if d.is_dir() and not d.name.startswith("__"):
            assert (d / "README.md").exists() or (d / "INDEX.md").exists(), f"{d.relative_to(ROOT)} has no README.md / INDEX.md"
