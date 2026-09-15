"""Portable plans and receipts. Importing this module never loads ML libraries."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

from .data import ARMS, digest, file_hash, read_rows

PACKAGES = {"ms-swift": "4.1.1", "transformers": "5.5.4", "peft": "0.18.1",
            "accelerate": "1.13.0", "safetensors": "0.7.0",
            "qwen-vl-utils": "0.0.14", "decord": "0.6.0"}


def package_versions():
    result = {name: importlib.metadata.version(name) for name in (
        *PACKAGES, "torch", "torchvision", "modelscope", "datasets", "trl", "tokenizers")}
    if any(result[name] != expected for name, expected in PACKAGES.items()):
        raise ValueError("package mismatch; use isolated documented environment")
    return result


def model_inventory(model):
    model = Path(model).resolve(strict=True)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5" or config.get("auto_map"):
        raise ValueError("local standard Qwen3.5 dense model required; no custom model code")
    if not (model / "tokenizer_config.json").is_file() or not (model / "tokenizer.json").is_file():
        raise ValueError("local tokenizer assets missing")
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("local safetensors weights missing")
    index = model / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if any(Path(name).name != name or not (model / name).is_file() for name in shards):
            raise ValueError("missing or escaping model shard")
    selected = {p for pattern in ("*.json", "*.jinja", "*.txt", "*.model", "*.safetensors")
                for p in model.glob(pattern) if p.is_file()}
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(selected)}
    return {"model_type": config["model_type"], "files": files, "tree_sha256": digest(files),
            "upstream_revision": None, "operator_must_confirm_4b": True}


def source_inventory(root):
    root = Path(root)
    files = [*sorted((root / "skill_annealing/rule_exposure").glob("*.py")),
             *sorted((root / "scripts").glob("rule_exposure*.py"))]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in files}


def environment(gpu, cache):
    if not re.fullmatch(r"0|[1-9][0-9]*", gpu):
        raise ValueError("one explicitly allocated GPU index required")
    env = os.environ.copy()
    if any(key in env for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NPROC_PER_NODE", "NNODES")):
        raise ValueError("use a plain single-process shell, not a distributed launch")
    inherited = env.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and inherited != gpu:
        raise ValueError("requested GPU conflicts with inherited allocation; do not override it")
    cache = Path(cache).resolve()
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1", WANDB_MODE="disabled",
               DO_NOT_TRACK="1", PYTHONUNBUFFERED="1", RULE_EXPOSURE_EXECUTE="1",
               HF_HOME=str(cache / "hf"), MODELSCOPE_CACHE=str(cache / "modelscope"),
               XDG_CACHE_HOME=str(cache / "xdg"), TORCH_HOME=str(cache / "torch"))
    return env


def training_command(job, root):
    config = job["config"]
    steps = config["train_cases"] * 4 // config["gradient_accumulation_steps"]
    arm = job["arm"]
    if arm not in ARMS:
        raise ValueError("unsupported arm")
    return [
        sys.executable, "-m", "swift.cli.sft",
        "--model", job["model"], "--model_type", "qwen3_5", "--template", "qwen3_5",
        "--tuner_type", "lora", "--dataset", str(Path(job["data"]) / (arm + ".jsonl")),
        "--split_dataset_ratio", "0", "--max_steps", str(steps), "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
        "--learning_rate", str(config["learning_rate"]), "--lora_rank", str(config["lora_rank"]),
        "--lora_alpha", str(config["lora_alpha"]), "--lora_dropout", "0",
        "--target_modules", "all-linear", "--freeze_llm", "false", "--freeze_vit", "true",
        "--freeze_aligner", "true", "--torch_dtype", "bfloat16", "--attn_impl", "sdpa",
        "--optim", "adamw_torch", "--weight_decay", "0", "--warmup_steps", "0",
        "--lr_scheduler_type", "linear", "--max_grad_norm", "1",
        "--max_length", str(config["max_length"]), "--truncation_strategy", "delete",
        "--strict", "true", "--lazy_tokenize", "true", "--packing", "false", "--padding_free", "false",
        "--group_by_length", "false", "--padding_side", "right",
        "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--dataloader_drop_last", "false", "--streaming", "false",
        "--enable_thinking", "false", "--add_non_thinking_prefix", "true", "--use_logits_to_keep", "true",
        "--gradient_checkpointing", "true", "--gradient_checkpointing_kwargs", '{"use_reentrant":false}',
        "--dataset_num_proc", "1", "--dataloader_num_workers", "0",
        "--load_from_cache_file", "false", "--load_args", "false", "--load_data_args", "false",
        "--save_strategy", "steps", "--save_steps", str(steps), "--save_total_limit", "1",
        "--save_only_model", "false", "--logging_steps", "1", "--report_to", "none",
        "--push_to_hub", "false", "--seed", str(config["seed"]), "--data_seed", str(config["seed"]),
        "--check_model", "false", "--external_plugins", str(Path(root) / "scripts/rule_exposure_callback.py"),
        "--callbacks", "rule_exposure_audit", "--output_dir", str(Path(job["stage_dir"]) / "training"),
    ]


def verify_checkpoint(stage, steps):
    paths = list((Path(stage) / "training").rglob(f"checkpoint-{steps}/trainer_state.json"))
    if len(paths) != 1:
        raise ValueError("fixed endpoint checkpoint missing or ambiguous")
    state = json.loads(paths[0].read_text(encoding="utf-8"))
    if state.get("global_step") != steps or state.get("max_steps") != steps:
        raise ValueError("training step budget mismatch")
    checkpoint = paths[0].parent
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError("saved adapter missing")
    return checkpoint


def adapter_hash(checkpoint):
    return digest({name: file_hash(Path(checkpoint) / name)
                   for name in ("adapter_config.json", "adapter_model.safetensors")})


def verify_predictions(rows, prompts, simulated):
    expected = {row["id"] for row in prompts}
    if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
        raise ValueError("predictions missing, duplicated, or from a different panel")
    for row in rows:
        if row.get("simulated") is not simulated or not isinstance(row.get("text"), str):
            raise ValueError("prediction backend/content mismatch")
        if not simulated:
            for key in ("prompt_tokens", "completion_tokens"):
                if type(row.get(key)) is not int or row[key] < 0:
                    raise ValueError("missing actual token usage")


def verify_stage(stage, job):
    stage = Path(stage)
    receipt = json.loads((stage / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("job_sha256") != digest(job) or receipt.get("status") != "completed":
        raise ValueError("invalid stage receipt")
    if job["kind"] == "train":
        steps = job["config"]["train_cases"] * 4 // job["config"]["gradient_accumulation_steps"]
        checkpoint = verify_checkpoint(stage, steps)
        audit = json.loads((stage / "training_audit.json").read_text(encoding="utf-8"))
        if audit["status"] != "completed" or audit["steps"] != list(range(1, steps + 1)):
            raise ValueError("incomplete training callback audit")
        if audit["initial_sha256"] == audit["final_sha256"]:
            raise ValueError("LoRA weights did not change")
        if job.get("expected_init") and audit["initial_sha256"] != job["expected_init"]:
            raise ValueError("unmatched LoRA initialization")
        if (receipt["initial_sha256"] != audit["initial_sha256"]
                or receipt["checkpoint"] != checkpoint.relative_to(stage).as_posix()):
            raise ValueError("training receipt does not match endpoint audit")
        if adapter_hash(checkpoint) != receipt["adapter_sha256"]:
            raise ValueError("adapter modified after completion")
    if job["kind"] == "infer":
        path = stage / "predictions.jsonl"
        if file_hash(path) != receipt["predictions_sha256"]:
            raise ValueError("prediction file changed after completion")
        verify_predictions(read_rows(path), read_rows(Path(job["data"]) / "eval_prompts.jsonl"), False)
    return receipt
