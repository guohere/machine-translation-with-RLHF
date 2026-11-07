"""Run machinery for the Phase 2 driver (scripts/p2_run.py).

Everything the Kaggle notebook used to do in its cells lives here, so the notebook
can stay a fixed launcher and changes arrive with `git pull`:
  * sh / run_jobs: run commands (two at a time on two GPUs), stream the output, keep a
    log per step in results/p2/<model>/logs/, and stop at the first failure;
  * save: push results/p2/ to the branch after each step (needs GH_TOKEN);
  * checks for what is already done, so every run continues where the last stopped.
"""

from __future__ import annotations

import base64
import glob
import gzip
import os
import shutil
import subprocess
import time

from mtrlhf.io import read_json, read_jsonl
from mtrlhf.models import CheckpointStore, model_paths


class StepFailed(RuntimeError):
    pass


class Runner:
    def __init__(self, model: str, branch: str, gh_token: str | None = None, hub_repo: str | None = None,
                 dry_run: bool = False, root: str = "."):
        self.model = model
        self.P = model_paths(model)
        self.branch = branch
        self.gh_token = gh_token
        self.dry_run = dry_run
        self.root = root
        self.res = f"results/p2/{model}"
        self.store = CheckpointStore.from_env(self.P["ckpt"], hub_repo, self.P["hub_prefix"])
        self.ran: list[str] = []  # steps that did real work in this run

    # --- running commands ----------------------------------------------------------------
    def sh(self, cmd: str, name: str) -> None:
        """Run a command, stream its output, keep it in <res>/logs/<name>.log; stop on failure."""
        self.ran.append(name)
        if self.dry_run:
            print(f"[dry-run] {name}: {cmd}", flush=True)
            return
        os.makedirs(f"{self.res}/logs", exist_ok=True)
        log_path = f"{self.res}/logs/{name}.log"
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            proc.wait()
        if proc.returncode != 0:
            self.fail(name, proc.returncode, log_path)

    def run_jobs(self, jobs: list[tuple[str, str]], report_every: int = 600) -> None:
        """Run (name, command) jobs two at a time on two GPUs if available, else one after another."""
        if not jobs:
            return
        if self.dry_run:
            for name, cmd in jobs:
                self.sh(cmd, name)
            return
        self.ran.extend(name for name, _ in jobs)
        os.makedirs(f"{self.res}/logs", exist_ok=True)
        slots = max(1, min(n_gpus(), 2))
        pending, running, failed, last = list(jobs), [], [], time.time()
        while pending or running:
            while pending and len(running) < slots:
                name, cmd = pending.pop(0)
                gpu = min(set(range(slots)) - {r[2] for r in running})
                log = open(f"{self.res}/logs/{name}.log", "w")
                proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
                running.append((name, proc, gpu, log))
                print(f"started {name} on GPU {gpu}", flush=True)
            time.sleep(15)
            for r in list(running):
                if r[1].poll() is not None:
                    r[3].close()
                    running.remove(r)
                    tail = open(f"{self.res}/logs/{r[0]}.log").read().strip().splitlines()[-15:]
                    print(f"{r[0]} finished (exit code {r[1].returncode}):", *tail, sep="\n  ", flush=True)
                    if r[1].returncode != 0:
                        failed.append((r[0], r[1].returncode))
            if running and time.time() - last > report_every:
                last = time.time()
                for name, *_ in running:
                    lines = open(f"{self.res}/logs/{name}.log").read().strip().splitlines()
                    print(f"[{time.strftime('%H:%M')}] {name}: {lines[-1][-160:] if lines else '(starting)'}",
                          flush=True)
        if failed:
            self.fail(", ".join(n for n, _ in failed), failed[0][1], f"{self.res}/logs/")

    def fail(self, name: str, code: int, log_path: str) -> None:
        self.save(f"Phase 2 ({self.model}): {name} FAILED, log in {log_path}")
        raise StepFailed(f"{name} failed (exit code {code}). The full output is above and in {log_path} on GitHub.")

    # --- saving to GitHub ------------------------------------------------------------------
    def _git(self, *args: str) -> tuple[int, str]:
        auth = base64.b64encode(f"x-access-token:{self.gh_token}".encode()).decode()
        cmd = ["git", "-C", self.root, "-c", "user.name=Phase 2 notebook",
               "-c", "user.email=noreply@users.noreply.github.com",
               "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}", *args]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        return p.returncode, out.replace(self.gh_token, "***").replace(auth, "***") if self.gh_token else out

    def collect(self) -> None:
        """Copy small artifacts from data/ and runs/ into results/p2/ (which is pushed)."""
        os.makedirs(self.res, exist_ok=True)
        md = self.P["model_data"]
        if os.path.exists("data/p2/data_stats.json"):
            shutil.copy("data/p2/data_stats.json", "results/p2/")
        for f in ("pairs_stats.json", "candidates.meta.json"):
            if os.path.exists(f"{md}/{f}"):
                shutil.copy(f"{md}/{f}", self.res)
        if os.path.exists(f"{md}/candidates.jsonl"):
            with open(f"{md}/candidates.jsonl", "rb") as src, gzip.open(f"{self.res}/candidates.jsonl.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        for run_dir in glob.glob(f"{self.P['runs']}/*"):
            for f in ("log_history.json", "samples.jsonl"):
                if os.path.exists(f"{run_dir}/{f}"):
                    os.makedirs(f"{self.res}/training/{os.path.basename(run_dir)}", exist_ok=True)
                    shutil.copy(f"{run_dir}/{f}", f"{self.res}/training/{os.path.basename(run_dir)}/")

    def save(self, message: str) -> None:
        if self.dry_run:
            print(f"[dry-run] save: {message}", flush=True)
            return
        self.collect()
        if not self.gh_token:
            return
        self._git("add", "results/p2")
        if self._git("diff", "--cached", "--quiet")[0] != 0:
            code, out = self._git("commit", "-q", "-m", message)
            if code:
                print("commit failed:", out)
                return
        code, out = self._git("pull", "-q", "--rebase", "origin", self.branch)
        if code:
            print("pull failed:", out)
            return
        code, out = self._git("push", "-q", "origin", f"HEAD:{self.branch}")
        print(f"saved to GitHub: {message}" if code == 0 else f"push failed: {out}", flush=True)

    def restore_candidates(self) -> None:
        """Bring back samples saved by an earlier run, together with the settings they used."""
        saved, meta, md = f"{self.res}/candidates.jsonl.gz", f"{self.res}/candidates.meta.json", self.P["model_data"]
        if os.path.exists(saved) and os.path.exists(meta) and not os.path.exists(f"{md}/candidates.jsonl"):
            os.makedirs(md, exist_ok=True)
            with gzip.open(saved, "rb") as src, open(f"{md}/candidates.jsonl", "wb") as dst:
                shutil.copyfileobj(src, dst)
            shutil.copy(meta, f"{md}/candidates.meta.json")
            print("restored sampled candidates from an earlier run")

    # --- what is already done ----------------------------------------------------------------
    def have(self, name: str) -> bool:
        return name in ("base", "opus-mt") or self.store.has(name)

    def candidates_done(self, sampling: dict, n: int, reward: str, n_prompts: int) -> bool:
        md = self.P["model_data"]
        meta_path = f"{md}/candidates.meta.json"
        if not os.path.exists(meta_path):
            return False
        meta = read_json(meta_path)
        same = (meta.get("temperature"), meta.get("top_p"), meta.get("n"), meta.get("reward")) == \
               (sampling["temperature"], sampling["top_p"], n, reward)
        return same and len(read_jsonl(f"{md}/candidates.jsonl")) >= n_prompts

    def pairs_sampling(self) -> dict | None:
        path = f"{self.P['model_data']}/pairs_stats.json"
        return read_json(path).get("sampling") if os.path.exists(path) else None

    def needs_training(self, name: str, data_sampling: dict | None = None) -> bool:
        """Missing, or trained on samples drawn with another sampling setting."""
        if not self.store.has(name):
            return True
        return data_sampling is not None and self.store.meta(name).get("data_sampling") != data_sampling

    def missing_outputs(self, system: str, testsets: list[str]) -> list[str]:
        """Test sets this system should have translated but has not (OPUS-MT only does en-de)."""
        from mtrlhf.models import EXTERNAL

        pairs = EXTERNAL.get(system, {}).get("pairs")
        wanted = [ts for ts in testsets if not pairs or tuple(ts.rsplit(".", 1)[-1].split("-")) in pairs]
        return [ts for ts in wanted if not os.path.exists(f"{self.P['outputs']}/{system}/{ts}.jsonl")]

    def drop_outputs(self, system: str) -> None:
        for f in glob.glob(f"{self.P['outputs']}/{system}/*.jsonl"):
            os.remove(f)


def n_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return 0
    return sum(line.startswith("GPU") for line in out.splitlines())
