"""A remote GPU box driven over ssh + rsync + tmux. Settings: the `box:` block of configs/backends/<name>.yaml.

    box = RemoteBox.from_backend("l40s")
    box.push_code(); box.serve("Qwen/Qwen1.5-1.8B", gpu=0); box.serve("AITeamVN/Vietnamese_Embedding", gpu=0, kind="embed"); box.run("configs/exp/B0_baseline.yaml"); box.pull()

Shared-box rules: only `box.root` and tmux sessions named `lawrag-*` are ever created or killed;
a GPU with less than `min_free_mib` free is refused.
"""
import re
import shlex
import time
import subprocess
from pathlib import Path
from typing import ClassVar

from config import Box, load_backend
from runs import RUNS, Ledger

PREFIX = "lawrag-"
MIN_RAM_GB, MIN_DISK_GB, MAX_RAM_GB = 24, 20, 32   # an M2 run holds docstore + FAISS (~10-15 GB RAM); 62 GB box shared with others
MAX_RUNS = 2  # LLM-bound runs on cached retrieval are light; never queue two index builds side by side
LABEL = "com.lawrag.autosync"  # launchd agent on the Mac
CODE_EXCLUDES = [".git", ".venv", ".rl_env", ".cache", "data", "__pycache__", ".pytest_cache", ".ruff_cache"]


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
    PORTS: ClassVar[dict[str, int]] = {"llm": 8000, "embed": 8100, "rerank": 8200}

    def serve(self, model: str, gpu: int = 0, kind: str = "llm") -> None:
        """vLLM server in tmux `lawrag-<kind>-g<gpu>`, port PORTS[kind] + gpu. kind: llm | embed | rerank.

        One GPU holds all three (gpu_share per kind); flags follow datamate (bf16, prefix caching, big batches)."""
        self.shell(self._serve_script(model, gpu, kind))

    def _serve_script(self, model: str, gpu: int, kind: str = "llm", replace: bool = False, share: float | None = None, tools: bool = False) -> str:
        """Shell that starts the server and waits until it answers (exit != 0 on failure).
        replace: kill our existing server on that GPU first (queue `serve` lines switch the LLM between runs)."""
        b, port, session = self.box, self.PORTS[kind] + gpu, f"{PREFIX}{kind}-g{gpu}"
        log = f"{b.root}/{RUNS}/logs/{kind}-g{gpu}.log"
        flags = b.llm_args if kind == "llm" else ["--runner", "pooling", *b.pooling_args]
        if tools:   # native tool calls (generate/react.py); Qwen3.5's chat template uses the qwen3_coder XML format
            flags = [*flags, "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
        server = (f"{self._env()} CUDA_VISIBLE_DEVICES={gpu} HF_HOME={b.hf_home} {b.vllm} serve {model} "
                  f"--host 127.0.0.1 --port {port} --seed 0 --gpu-memory-utilization {share or b.gpu_share[kind]} "
                  f"{' '.join(flags)} 2>&1 | tee {log}")
        kill = (f"tmux kill-session -t {session} 2>/dev/null && sleep 20 || true" if replace else
                f'tmux has-session -t {session} 2>/dev/null && {{ echo "already running: {session}"; exit 0; }}')
        return f"""
            {kill}
            {self._free_check(gpu, kind)}
            mkdir -p {b.root}/{RUNS}/logs
            tmux new -d -s {session} {shlex.quote(server)}
            for i in $(seq 1 120); do
              curl -sf localhost:{port}/v1/models >/dev/null && {{ echo "ready: {kind} {model} on GPU {gpu}, port {port}"; exit 0; }}
              tmux has-session -t {session} 2>/dev/null || {{ echo "server died, see {log}"; tail -5 {log}; exit 1; }}
              sleep 5
            done
            echo "not ready after 10 min, see {log}"; exit 1"""

    def _job(self, exp: str, gpu: int = 0, args: list[str] = ()) -> tuple[str, str]:
        """(tmux session name, shell command) of one experiment, or of a shell command: exp "sh", args [name, *cmd]."""
        b, name = self.box, Path(exp).stem
        prefix = (f"cd {b.root} && export {self._env()} PATH=$HOME/.local/bin:$PATH HF_HOME={b.hf_home} CUDA_VISIBLE_DEVICES={gpu} && "
                  f"PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax={MAX_RAM_GB}G ")  # cgroup cap: OOM-kills this run, not the box
        if exp == "sh":
            name, cmd = args[0], " ".join(args[1:])
            return f"{PREFIX}run-{name}", f"{prefix}bash -c {shlex.quote(cmd)} 2>&1 | tee {RUNS}/logs/{name}.log"
        sets = [a.split("=", 1)[1] for a in args if "=" in a and "base_url" not in a]  # e.g. split=test -> -test
        if sets:
            name += "-" + "-".join(re.sub(r"[^\w.]+", "_", v) for v in sets)[:40]
        argv = " ".join(shlex.quote(a) for a in args)
        job = f"{prefix}uv run python -m cli run {shlex.quote(exp)} {argv} 2>&1 | tee {RUNS}/logs/{name}.log"
        return f"{PREFIX}run-{name}-g{gpu}", job   # per-GPU suffix: same-named runs on both GPUs no longer collide

    def run(self, exp: str, gpu: int = 0, args: list[str] = ()) -> None:
        """One experiment in tmux `lawrag-run-<exp>`; the embedder / reranker use `gpu`; the LLM server must be up."""
        session, job = self._job(exp, gpu, args)
        self.shell(f"{self._guard()} && mkdir -p {self.box.root}/{RUNS}/logs && tmux new -d -s {session} {shlex.quote(job)} "
                   f"&& echo started {session}")

    def queue(self, jobs: list[list[str]], gpu: int = 0, parallel: bool = False) -> None:
        """Run experiments one after another ON THE BOX (tmux `lawrag-queue`), so this machine can be switched off.
        jobs = [[exp, *args]]; each waits for the previous lawrag-run-* to end and for the guard to pass.
        Other line kinds: ["sh", name, *cmd] runs a shell command as a run (guard + cgroup cap); ["wait", *cond] /
        ["do", *cmd] run inline. A job ["serve", model, "--gpu", "1"] swaps our LLM server on that GPU (waits for runs first); if it fails
        the queue stops, so later runs never hit a missing server."""
        b = self.box
        lines = ["set -u", f"cd {b.root}", f"mkdir -p {RUNS}/logs"]
        wait = "while [ -n \"${prev:-}\" ] && tmux has-session -t \"$prev\" 2>/dev/null; do sleep 60; done"  # own previous run
        for exp, *args in jobs:
            if exp in ("wait", "do"):   # inline in the queue script, not a run: "wait <condition>", "do <command>"
                cmd = " ".join(args)
                lines += [f"until {cmd}; do sleep 60; done" if exp == "wait" else f"{cmd} || true"]
                continue
            if exp == "serve":
                g = int(args[args.index("--gpu") + 1]) if "--gpu" in args else gpu
                share = float(args[args.index("--share") + 1]) if "--share" in args else None   # e.g. a 9B teacher alone on a GPU
                script = self._serve_script(args[0], g, replace=True, share=share, tools="--tools" in args)
                lines += [wait, f"( {script} ) || {{ echo 'serve failed, queue stopped'; exit 1; }}"]
                continue
            session, job = self._job(exp, gpu, args)
            lines += [wait,
                      f"until ( {self._guard()} ); do sleep 120; done",
                      f"tmux new -d -s {session} {shlex.quote(job)} && prev={session} && echo \"$(date +%F' '%R) started {session}\"",
                      "sleep 5"]
        lines += [wait, "echo queue done"]
        # a queue already running is not killed: this one waits for it (sessions lawrag-queue-<time>)
        tag = time.strftime("%m%d-%H%M%S")
        session, log = f"{PREFIX}queue-{tag}", f"{b.root}/{RUNS}/logs/queue-{tag}.log"
        if not parallel:  # default: start after the queues already running
            lines.insert(1, f"while tmux ls 2>/dev/null | grep '^{PREFIX}queue' | grep -vq '^{session}:'; do sleep 60; done")
        script = f"{b.root}/{RUNS}/logs/queue-{tag}.sh"
        self.shell(f"cat > {script} <<'QUEUE'\n" + "\n".join(lines) + "\nQUEUE\n"
                   f"tmux new -d -s {session} 'bash {script} 2>&1 | tee {log}' && echo queued {len(jobs)} jobs: {session}, log {log}")

    def _guard(self) -> str:
        """Shell test run before every launch: enough free RAM / disk and no other run of ours, else refuse.
        2026-11-18: 4 parallel M2 retrieval jobs + an index build hung the box (RAM) until a hard reboot."""
        return (f"ram=$(free -g | awk '/^Mem:/{{print $7}}'); disk=$(df -BG --output=avail {self.box.root} | tail -1 | tr -dc 0-9); "
                f"runs=$(tmux ls 2>/dev/null | grep -c '^{PREFIX}run-' || true); "
                f"echo \"box: RAM available ${{ram}}G, disk free ${{disk}}G, our runs ${{runs}}\"; "
                f"[ \"$ram\" -ge {MIN_RAM_GB} ] && [ \"$disk\" -ge {MIN_DISK_GB} ] && [ \"$runs\" -lt {MAX_RUNS} ] "
                f"|| {{ echo 'refused: need RAM >= {MIN_RAM_GB}G, disk >= {MIN_DISK_GB}G, fewer than {MAX_RUNS} lawrag-run'; exit 1; }}")

    def status(self) -> None:
        self.shell(f"""
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
            echo; tmux ls 2>/dev/null | grep {PREFIX} || echo "no {PREFIX}* sessions"
            echo; tail -5 {self.box.root}/{RUNS}/results.tsv 2>/dev/null | cut -f1-11 || true""")

    def stop(self, name: str) -> None:
        """Kill one of OUR sessions, e.g. stop("llm-g0") or stop("run-B0_baseline")."""
        self.shell(f"tmux kill-session -t {PREFIX}{name}")

    def tunnel(self) -> None:
        """Mac -> box: llm 8000/8001, embed 8100/8101, rerank 8200/8201 (blocks until Ctrl-C)."""
        fwd = [a for p in (8000, 8001, 8100, 8101, 8200, 8201) for a in ("-L", f"{p}:localhost:{p}")]
        self._local(["ssh", "-N", *fwd, self.box.ssh])

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
        """Run a bash script on the box (non-login: a login shell's .bash_logout breaks exit codes; stops at the first error)."""
        return self._local(["ssh", self.box.ssh, f"bash -c {shlex.quote('set -e; ' + script)}"], check=check)

    def _env(self) -> str:
        """`box.env` as KEY=VALUE words, CUDA_HOME dropped (like datamate); the dirs it names are created first."""
        env = {k: v.format(root=self.box.root) for k, v in self.box.env.items()}
        dirs = " ".join(shlex.quote(v) for k, v in env.items() if k.endswith(("_ROOT", "_BASE", "TMPDIR")))
        if dirs:
            self.shell(f"mkdir -p {dirs}")
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " CUDA_HOME="

    def _free_check(self, gpu: int, kind: str = "llm") -> str:
        """The LLM needs min_free_mib; a pooling server needs its share of the card (+1 GiB)."""
        q = f"nvidia-smi --format=csv,noheader,nounits -i {gpu} --query-gpu"
        need = (str(self.box.min_free_mib) if kind == "llm"
                else f"$(( $({q}=memory.total) * {int(self.box.gpu_share[kind] * 100)} / 100 + 1024 ))")
        return (f"free=$({q}=memory.free); need={need}; "
                f'[ "$free" -ge "$need" ] || {{ echo "GPU {gpu}: $free MiB free < $need needed, refusing"; exit 3; }}')

    @staticmethod
    def _local(cmd: list[str], check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, check=check, capture_output=quiet)
