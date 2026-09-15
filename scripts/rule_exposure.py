"""One entry: prepare, CPU simulate, bounded GPU run/resume, summarize and status."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from skill_annealing.rule_exposure.data import (
    ARMS, canonical, digest, file_hash, prepare, read_rows, validate_config, verify_data, write_json, write_rows)
from skill_annealing.rule_exposure.analysis import summarize
from skill_annealing.rule_exposure.runtime import environment, source_inventory, verify_stage

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    write_json(temp, value)
    temp.replace(path)


def prepare_work(work, config):
    work = Path(work)
    if work.exists() or work.is_symlink():
        raise FileExistsError("use a new work directory; run reuses verified existing data")
    work.mkdir(parents=True)
    return prepare(work / "data", config)


def simulate(work, config):
    prepare_work(work, config)
    work = Path(work)
    directory = work / "simulation"
    directory.mkdir()
    refs = read_rows(work / "data/eval_reference.jsonl")
    for endpoint in ("base", *ARMS):
        # An oracle round trip, deliberately tagged; not a fake trained model.
        write_rows(directory / (endpoint + ".jsonl"), [
            {"id": row["id"], "text": canonical({"decision": row["target"]}),
             "simulated": True, "prompt_tokens": None, "completion_tokens": None, "seconds": None}
            for row in refs])
    return summarize(work, simulated=True)


@contextmanager
def gpu_lock(gpu):
    import fcntl
    directory = ROOT / "tmp"
    directory.mkdir(exist_ok=True)
    with (directory / ("rule_exposure_gpu_" + gpu + ".lock")).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another pilot controller in this checkout owns this GPU")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def idle_gpu(gpu):
    result = subprocess.run(["nvidia-smi", "--id=" + gpu,
                             "--query-gpu=name,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("exactly one allocated GPU required")
    name, used, utilization = [x.strip() for x in rows[0].split(",")]
    if int(used) > 2048 or int(utilization) > 5:
        raise RuntimeError("allocated GPU appears busy; no process will be killed")
    return {"name": name, "memory_used_mib": int(used), "utilization_percent": int(utilization)}


def launch(command, env, log_path, timeout):
    """Own one process group; terminate only that group on timeout/interruption."""
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, shell=False)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            raise
        if result:
            raise RuntimeError("worker failed; inspect the local stage log (not uploaded)")


def run(work, config, model, gpu, execute=False, retry_failed=False):
    work, model = Path(work).resolve(), Path(model).resolve()
    if work.is_relative_to(model) or model.is_relative_to(work):
        raise ValueError("model and work directories must be disjoint")
    if not work.exists():
        prepare_work(work, config)
    if (work / "simulation").exists():
        raise ValueError("simulation directories cannot host a real campaign; use a new directory")
    manifest = verify_data(work / "data")
    if manifest["config"] != config:
        raise ValueError("existing work has a different config; use a new directory")
    plan = {"status": "dry_run", "arms": list(ARMS), "training_runs": 4,
            "updates_per_run": manifest["optimizer_steps"],
            "eval_endpoints": 5, "generations": manifest["eval_prompts_per_endpoint"] * 5,
            "max_campaign_seconds": config["campaign_timeout_seconds"], "gpu_used": False}
    if not execute:
        return plan
    if os.name != "posix":
        raise ValueError("real execution requires Linux; CPU simulation is portable")
    if not model.is_dir():
        raise ValueError("complete local model required; no automatic download")
    env = environment(gpu, work / "cache")
    env["PYTHONHASHSEED"] = str(config["seed"])
    identity = {"config_sha256": digest(config), "data_sha256": file_hash(work / "data/manifest.json"),
                "source_sha256": source_inventory(ROOT), "model_path": str(model), "gpu": gpu}
    state_path = work / "campaign.json"
    with gpu_lock(gpu):
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["identity"] != identity:
                raise ValueError("code/data/config/model location/GPU changed; use a new campaign")
        else:
            state = {"status": "prepared", "identity": identity, "spent_seconds": 0, "stages": {},
                     "gpu": idle_gpu(gpu), "prior_failed_attempts": []}
            atomic_json(state_path, state)
        # Unclean interruption consumes its full reserved timeout; never grants free retries.
        for entry in state["stages"].values():
            if entry["status"] == "running":
                state["spent_seconds"] += entry["reserved_seconds"]
                entry["status"] = "interrupted"
        atomic_json(state_path, state)
        preflight = None
        initial = None
        adapters = {}
        schedule = [("preflight", "preflight", "base")]
        schedule += [("train_" + arm, "train", arm) for arm in ARMS]
        schedule += [("infer_" + arm, "infer", arm) for arm in ("base", *ARMS)]
        for key, kind, arm in schedule:
            previous = state["stages"].get(key)
            if previous and previous["status"] == "completed":
                stage = work / previous["directory"]
                job = json.loads((stage / "job.json").read_text(encoding="utf-8"))
                receipt = verify_stage(stage, job)
                if file_hash(stage / "receipt.json") != previous["receipt_sha256"]:
                    raise ValueError("completed stage receipt changed")
            else:
                if previous and not retry_failed:
                    raise RuntimeError("failed/interrupted stage preserved; fix environment then explicitly --retry-failed")
                attempt = 1 if previous is None else previous["attempt"] + 1
                if attempt > 2:
                    raise RuntimeError("at most two attempts per stage; no automatic quality retries")
                remaining = config["campaign_timeout_seconds"] - state["spent_seconds"]
                if remaining <= 0:
                    raise RuntimeError("campaign wall-time budget exhausted")
                idle_gpu(gpu)
                stage = work / "stages" / (key + f"_attempt{attempt}")
                stage.mkdir(parents=True, exist_ok=False)
                job = {"kind": kind, "arm": arm, "config": config, "data": str(work / "data"),
                       "data_sha256": identity["data_sha256"], "model": str(model), "stage_dir": str(stage)}
                if preflight:
                    job.update(packages=preflight["packages"],
                               model_stat_signature=preflight["model_stat_signature"],
                               model_tree_sha256=preflight["model_inventory"]["tree_sha256"])
                if kind == "train" and initial:
                    job["expected_init"] = initial
                if kind == "infer" and arm != "base":
                    job.update(adapter=adapters[arm][0], adapter_sha256=adapters[arm][1])
                write_json(stage / "job.json", job)
                if previous:
                    state["prior_failed_attempts"].append({"key": key, **previous})
                reserved = min(remaining, config["stage_timeout_seconds"])
                entry = {"status": "running", "attempt": attempt,
                         "directory": stage.relative_to(work).as_posix(), "reserved_seconds": reserved}
                state["stages"][key] = entry
                state["status"] = "running"
                atomic_json(state_path, state)
                print(f"Starting {key}, attempt {attempt}", flush=True)
                start = time.monotonic()
                try:
                    launch([sys.executable, "-B", "-m", "scripts.rule_exposure_worker",
                            "--job", str(stage / "job.json")], env, stage / "worker.log", reserved)
                    receipt = verify_stage(stage, job)
                    entry.update(status="completed", receipt_sha256=file_hash(stage / "receipt.json"))
                except BaseException:
                    entry["status"] = "failed"
                    state["status"] = "incomplete"
                    raise
                finally:
                    state["spent_seconds"] += time.monotonic() - start
                    atomic_json(state_path, state)
            if kind == "preflight":
                preflight = receipt
                # Check immutable model signature even when all stages are skipped on resume.
                for name, values in receipt["model_stat_signature"].items():
                    stat = (model / name).stat()
                    if [stat.st_size, stat.st_mtime_ns] != values:
                        raise ValueError("model changed since preflight")
            elif kind == "train":
                if initial is None:
                    initial = receipt["initial_sha256"]
                elif initial != receipt["initial_sha256"]:
                    raise ValueError("paired initialization differs")
                adapters[arm] = (str(stage / receipt["checkpoint"]), receipt["adapter_sha256"])
        state["status"] = "completed"
        atomic_json(state_path, state)
        return summarize(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "simulate", "run", "summarize", "status"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rule_exposure_pilot.json")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        if args.model is None or args.gpu is None:
            parser.error("run requires --model and --gpu, including dry-run")
        result = run(args.workdir, load_config(args.config), args.model, args.gpu, args.execute, args.retry_failed)
    elif args.action == "prepare":
        result = prepare_work(args.workdir, load_config(args.config))
    elif args.action == "simulate":
        result = simulate(args.workdir, load_config(args.config))
    elif args.action == "summarize":
        result = summarize(args.workdir)
    else:
        path = args.workdir / "campaign.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        result = {"status": state.get("status", "not_started"),
                  "spent_seconds": state.get("spent_seconds", 0),
                  "stages": {k: v["status"] for k, v in state.get("stages", {}).items()}}
    print(json.dumps({k: v for k, v in result.items() if k not in ("metrics", "input_sha256", "files")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
