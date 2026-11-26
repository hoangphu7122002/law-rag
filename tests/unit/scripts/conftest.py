"""The scripts are not a package: put every scripts/<area>/ directory on sys.path so the tests can `import cascade_stats` etc."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "scripts"
for area in ("analysis", "data", "train", "serve", "ops"):
    sys.path.insert(0, str(ROOT / area))
