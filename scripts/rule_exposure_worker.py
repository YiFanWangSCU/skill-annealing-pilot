"""Private GPU worker launched by the bounded pilot controller, not by pytest."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from skill_annealing.rule_exposure.data import ARMS, digest, file_hash, read_rows, verify_data, write_json, canonical
from skill_annealing.rule_exposure.runtime import (
    adapter_hash, model_inventory, package_versions, training_command, verify_checkpoint, verify_predictions)


def preflight(job):
    from swift import get_model_processor, get_template
    inventory = model_inventory(job["model"])
    _, processor = get_model_processor(job["model"], model_type="qwen3_5",
                                       load_model=False, download_model=False)
    config = job["config"]
    template = get_template(processor, template_type="qwen3_5", max_length=config["max_length"],
                            truncation_strategy="raise", add_non_thinking_prefix=True,
                            enable_thinking=False)
    stats = {}
    template.set_mode("train")
    for arm in ARMS:
        lengths, supervised = [], []
        for row in read_rows(Path(job["data"]) / (arm + ".jsonl")):
            encoded = template.encode(copy.deepcopy(row))
            lengths.append(len(encoded["input_ids"]))
            supervised.append(sum(label != -100 for label in encoded["labels"]))
        if min(supervised) <= 0:
            raise ValueError("no supervised target tokens")
        stats[arm] = {"records": len(lengths), "max_tokens": max(lengths),
                      "mean_tokens": sum(lengths) / len(lengths), "total_tokens": sum(lengths)}
    template.set_mode("transformers")
    for row in read_rows(Path(job["data"]) / "eval_prompts.jsonl"):
        n = len(template.encode({"messages": copy.deepcopy(row["messages"])})["input_ids"])
        if n + config["max_new_tokens"] > config["max_length"]:
            raise ValueError("evaluation prompt plus output exceeds length budget")
    # Stat signature permits cheap mutation checks on resume; exact hashes are retained.
    signature = {name: [Path(job["model"], name).stat().st_size, Path(job["model"], name).stat().st_mtime_ns]
                 for name in inventory["files"]}
    return {"model_inventory": inventory, "model_stat_signature": signature,
            "training_token_lengths": stats, "tokenization_checked": True}


def train(job, job_path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["RULE_EXPOSURE_JOB"] = str(job_path.resolve())
    subprocess.run(training_command(job, root), check=True, env=env, shell=False)
    steps = job["config"]["train_cases"] * 4 // job["config"]["gradient_accumulation_steps"]
    checkpoint = verify_checkpoint(job["stage_dir"], steps)
    audit = json.loads((Path(job["stage_dir"]) / "training_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "completed":
        raise ValueError("training callback did not finish")
    return {"checkpoint": checkpoint.relative_to(Path(job["stage_dir"])).as_posix(),
            "adapter_sha256": adapter_hash(checkpoint),
            "initial_sha256": audit["initial_sha256"], "optimizer_steps": steps}


def infer(job):
    import torch
    from swift import get_model_processor, get_template
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.tuners import Swift
    torch.manual_seed(job["config"]["seed"])
    model, processor = get_model_processor(job["model"], model_type="qwen3_5",
                                           torch_dtype=torch.bfloat16, device_map="cuda:0",
                                           attn_impl="sdpa", download_model=False)
    if job.get("adapter"):
        if adapter_hash(job["adapter"]) != job["adapter_sha256"]:
            raise ValueError("adapter identity changed")
        model = Swift.from_pretrained(model, job["adapter"], is_trainable=False)
    template = get_template(processor, template_type="qwen3_5", max_length=job["config"]["max_length"],
                            truncation_strategy="raise", enable_thinking=False)
    model.eval()
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    prompts = read_rows(Path(job["data"]) / "eval_prompts.jsonl")  # No reference labels read here.
    predictions = []
    path = Path(job["stage_dir"]) / "predictions.jsonl"
    request = RequestConfig(max_tokens=job["config"]["max_new_tokens"], temperature=0,
                            top_p=1, stream=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in prompts:
            torch.cuda.synchronize()
            start = time.perf_counter()
            response = engine.infer([InferRequest(messages=copy.deepcopy(row["messages"]))],
                                    request, use_tqdm=False)[0]
            torch.cuda.synchronize()
            prediction = {"id": row["id"], "text": response.choices[0].message.content,
                          "prompt_tokens": response.usage.prompt_tokens,
                          "completion_tokens": response.usage.completion_tokens,
                          "seconds": time.perf_counter() - start, "simulated": False}
            predictions.append(prediction)
            stream.write(canonical(prediction) + "\n")
            stream.flush()
    verify_predictions(predictions, prompts, False)
    return {"predictions_sha256": file_hash(path), "prediction_count": len(predictions),
            "timing_scope": "single_query_inference_only_no_loading_no_warmup_exclusion"}


def run(job_path):
    if os.environ.get("RULE_EXPOSURE_EXECUTE") != "1":
        raise ValueError("use the explicit controller --execute entry")
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    data = verify_data(job["data"])
    if data["config"] != job["config"] or file_hash(Path(job["data"]) / "manifest.json") != job["data_sha256"]:
        raise ValueError("job data contract mismatch")
    packages = package_versions()
    if job.get("packages") and packages != job["packages"]:
        raise ValueError("runtime changed between stages")
    if not job["kind"] == "preflight":
        for name, values in job["model_stat_signature"].items():
            stat = (Path(job["model"]) / name).stat()
            if [stat.st_size, stat.st_mtime_ns] != values:
                raise ValueError("model assets changed between stages")
    if job["kind"] == "preflight":
        result = preflight(job)
    elif job["kind"] == "train":
        result = train(job, Path(job_path))
    elif job["kind"] == "infer":
        result = infer(job)
    else:
        raise ValueError("unknown worker stage")
    result.update(status="completed", job_sha256=digest(job), packages=packages)
    write_json(Path(job["stage_dir"]) / "receipt.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    run(parser.parse_args().job)
