"""A remote GPU box driven over ssh + rsync + tmux. Settings: the `box:` block of configs/backends/<name>.yaml.

    box = RemoteBox.from_backend("l40s")
    box.push_code(); box.serve("Qwen/Qwen1.5-1.8B", gpu=0); box.run("configs/exp/B0_baseline.yaml"); box.pull()

Shared-box rules: only `box.root` and tmux sessions named `lawrag-*` are ever created or killed;
a GPU with less than `min_free_mib` free is refused.
"""
import shlex
import subprocess
from pathlib import Path

from config import Box, load_backend
from runs import RUNS, Ledger

PREFIX = "lawrag-"
LABEL = "com.lawrag.autosync"  # launchd agent on the Mac
CODE_EXCLUDES = [".git", ".venv", ".cache", "data", "__pycache__", ".pytest_cache", ".ruff_cache"]


class RemoteBox:
    def __init__(self, box: Box):
        self.box = box

    @classmethod
    def from_backend(cls, name: str = "l40s") -> "RemoteBox":
        backend = load_backend(name)
        if backend.box is None:
            raise ValueError(f"backend {name!r} has no `box:` block")
        return cls(backend.box)

    # ---- files ----
    def push_code(self) -> None:
        self.shell(f"mkdir -p {self.box.root}")
        excludes = [f"--exclude={e}" for e in CODE_EXCLUDES]
        self._local(["rsync", "-az", "--delete", *excludes, "./", f"{self.box.ssh}:{self.box.root}/"])

    def push_data(self) -> None:
        """Raw inputs only; the box rebuilds data/processed with `make data` (see setup)."""
        self.shell(f"mkdir -p {self.box.root}/data/raw")
        self._local(["rsync", "-az", "--progress", "data/raw/", f"{self.box.ssh}:{self.box.root}/data/raw/"])

    def watch(self, every: float = 2.0) -> None:
        """Push code whenever a tracked file changes (polls mtimes; blocks). `autosync on` runs this under launchd."""
        import time

        last = None
        while True:
            stamp = self._code_stamp()
            if stamp != last:
                ok = self._push_quiet()
                print(time.strftime("%H:%M:%S"), "pushed" if ok else "push failed (box offline?)", flush=True)
                last = stamp if ok else None
            time.sleep(every)

    def autosync(self, on: bool) -> None:
        """Install / remove a launchd agent (macOS) that runs `watch` from login on."""
        import os
        import sys

        plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
        subprocess.run(["launchctl", "unload", str(plist)], check=False, capture_output=True)
        if not on:
            plist.unlink(missing_ok=True)
            print("autosync off")
            return
        repo, log = Path.cwd(), Path.cwd() / "data/runs/logs/autosync.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        args = "".join(f"<string>{a}</string>" for a in [sys.executable, "-m", "cli", "box", "watch"])
        plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{LABEL}</string>
<key>ProgramArguments</key><array>{args}</array>
<key>WorkingDirectory</key><string>{repo}</string>
<key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>src</string>
<key>PATH</key><string>{os.environ.get("PATH", "/usr/bin:/bin")}</string></dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardOutPath</key><string>{log}</string><key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
""")
        subprocess.run(["launchctl", "load", str(plist)], check=True)
        print(f"autosync on: {repo} -> {self.box.ssh}:{self.box.root} (log {log})")

    def pull(self) -> None:
        """Run folders back to the Mac; the box ledger rows are merged into the local one."""
        RUNS.mkdir(parents=True, exist_ok=True)
        remote = f"{self.box.ssh}:{self.box.root}/{RUNS}"
        self._local(["rsync", "-az", "--exclude=results.tsv", f"{remote}/", f"{RUNS}/"])
        tmp = RUNS / ".results_box.tsv"
        if self._local(["rsync", "-az", f"{remote}/results.tsv", str(tmp)], check=False).returncode == 0:
            print(f"ledger: +{Ledger().merge(tmp)} rows from the box")
            tmp.unlink()

    # ---- environment ----
    def setup(self) -> None:
        """One time: uv in the user's home, project env, processed data, tests."""
        self.shell(f"""
            command -v uv >/dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH=$HOME/.local/bin:$PATH
            cd {self.box.root} && uv sync --extra models --extra dev
            [ -f data/processed/qa.parquet ] || make data
            make test""")

    # ---- GPU jobs ----
    def serve(self, model: str, gpu: int = 0) -> None:
        """vLLM OpenAI-compatible server on one GPU, port 8000 + gpu, in tmux `lawrag-vllm-g<gpu>`."""
        b, port, session = self.box, 8000 + gpu, f"{PREFIX}vllm-g{gpu}"
        log = f"{b.root}/{RUNS}/logs/vllm-g{gpu}.log"
        server = (f"{self._env()} CUDA_VISIBLE_DEVICES={gpu} HF_HOME={b.hf_home} {b.vllm} serve {model} --port {port} --seed 0 "
                  f"--gpu-memory-utilization {b.gpu_memory_utilization} --max-model-len {b.max_model_len} 2>&1 | tee {log}")
        self.shell(f"""
            {self._free_check(gpu)}
            tmux has-session -t {session} 2>/dev/null && {{ echo "already running: {session}"; exit 0; }}
            mkdir -p {b.root}/{RUNS}/logs
            tmux new -d -s {session} {shlex.quote(server)}
            for i in $(seq 1 120); do
              curl -sf localhost:{port}/v1/models >/dev/null && {{ echo "ready: {model} on GPU {gpu}, port {port}"; exit 0; }}
              sleep 5
            done
            echo "not ready after 10 min, see {log}"; exit 1""")

    def run(self, exp: str, gpu: int = 0, args: list[str] = ()) -> None:
        """One experiment in tmux `lawrag-run-<exp>`; the embedder / reranker use `gpu`; the LLM server must be up."""
        b, name = self.box, Path(exp).stem
        argv = " ".join(shlex.quote(a) for a in args)
        job = (f"cd {b.root} && export {self._env()} PATH=$HOME/.local/bin:$PATH HF_HOME={b.hf_home} CUDA_VISIBLE_DEVICES={gpu} && "
               f"PYTHONPATH=src uv run python -m cli run {shlex.quote(exp)} {argv} 2>&1 | tee {RUNS}/logs/{name}.log")
        self.shell(f"mkdir -p {b.root}/{RUNS}/logs && tmux new -d -s {PREFIX}run-{name} {shlex.quote(job)} "
                   f"&& echo started {PREFIX}run-{name}")

    def status(self) -> None:
        self.shell(f"""
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
            echo; tmux ls 2>/dev/null | grep {PREFIX} || echo "no {PREFIX}* sessions"
            echo; tail -5 {self.box.root}/{RUNS}/results.tsv 2>/dev/null | cut -f1-11 || true""")

    def stop(self, name: str) -> None:
        """Kill one of OUR sessions, e.g. stop("vllm-g0") or stop("run-B0_baseline")."""
        self.shell(f"tmux kill-session -t {PREFIX}{name}")

    def tunnel(self) -> None:
        """Mac -> box: localhost:8000/8001 reach the vLLM servers (blocks until Ctrl-C)."""
        self._local(["ssh", "-N", "-L", "8000:localhost:8000", "-L", "8001:localhost:8001", self.box.ssh])

    # ---- plumbing ----
    def _code_stamp(self) -> tuple:
        """Newest mtime + file count of everything push_code would send."""
        import os

        newest, n = 0.0, 0
        for folder, dirs, names in os.walk("."):
            dirs[:] = [d for d in dirs if d not in CODE_EXCLUDES]
            for name in names:
                newest, n = max(newest, os.stat(os.path.join(folder, name)).st_mtime), n + 1
        return newest, n

    def _push_quiet(self) -> bool:
        excludes = [f"--exclude={e}" for e in CODE_EXCLUDES]
        cmd = ["rsync", "-az", "--delete", "-e", "ssh -o ConnectTimeout=10 -o BatchMode=yes", *excludes,
               "./", f"{self.box.ssh}:{self.box.root}/"]
        return self._local(cmd, check=False, quiet=True).returncode == 0

    def shell(self, script: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a bash script on the box (login shell, stops at the first error)."""
        return self._local(["ssh", self.box.ssh, f"bash -lc {shlex.quote('set -e; ' + script)}"], check=check)

    def _env(self) -> str:
        """`box.env` as KEY=VALUE words, CUDA_HOME dropped (like datamate); the dirs it names are created first."""
        env = {k: v.format(root=self.box.root) for k, v in self.box.env.items()}
        dirs = " ".join(shlex.quote(v) for k, v in env.items() if k.endswith(("_ROOT", "_BASE", "TMPDIR")))
        if dirs:
            self.shell(f"mkdir -p {dirs}")
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " CUDA_HOME="

    def _free_check(self, gpu: int) -> str:
        return (f"free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {gpu}); "
                f'[ "$free" -ge {self.box.min_free_mib} ] || {{ echo "GPU {gpu} has only $free MiB free, refusing"; exit 3; }}')

    @staticmethod
    def _local(cmd: list[str], check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, check=check, capture_output=quiet)
