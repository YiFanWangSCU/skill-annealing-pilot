"""Case-paired descriptive analysis; one seed is not a confirmatory method result."""
import csv
import json
from pathlib import Path
import random
import re
from statistics import mean

from .data import ARMS, DECISIONS, file_hash, read_rows, verify_data, write_json
from .runtime import verify_predictions, verify_stage


def parse(text):
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text).strip()
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result
        obj = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"decision"}:
        return None
    return obj["decision"] if obj["decision"] in DECISIONS else None


def paired_interval(reference, scores_a, scores_b, *, seed, resamples):
    # Both unseen-value policies stay in the same case cluster.
    by_case = {}
    for row in reference:
        if row["view"] == "short" and row["policy_category"] == "unseen_values":
            by_case.setdefault(row["case_id"], []).append(scores_a[row["id"]] - scores_b[row["id"]])
    differences = [mean(v) for v in by_case.values()]
    if not differences:
        return {"mean_pp": None, "ci95_pp": None, "cases": 0}
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(differences, k=len(differences))) * 100 for _ in range(resamples))
    return {"mean_pp": mean(differences) * 100,
            "ci95_pp": [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]],
            "cases": len(differences), "scope": "case_resampling_conditional_on_one_seed_and_fixed_policy_versions"}


def score_endpoint(reference, predictions):
    lookup = {r["id"]: r for r in predictions}
    scores = {row["id"]: int(parse(lookup[row["id"]]["text"]) == row["target"]) for row in reference}
    metrics = {}
    for view in ("full", "short"):
        rows = [row for row in reference if row["view"] == view]
        base_rows = {row["case_id"]: row for row in rows if row["policy_index"] == 0}
        def accuracy(selected):
            return mean(scores[r["id"]] for r in selected) if selected else None
        changed = [r for r in rows if r["affected"]]
        unchanged = [r for r in rows if r["policy_index"] > 0 and not r["affected"]]
        old_correct_unchanged = [r for r in unchanged if scores[base_rows[r["case_id"]]["id"]]]
        paired = [r for r in rows if r["policy_index"] > 0]
        both = [scores[r["id"]] * scores[base_rows[r["case_id"]]["id"]] for r in paired]
        token_values = [lookup[r["id"]]["prompt_tokens"] for r in rows]
        metrics[view] = {
            "accuracy": accuracy(rows), "rows": len(rows),
            "base_accuracy": accuracy([r for r in rows if r["policy_category"] == "base"]),
            "unseen_combination_accuracy": accuracy([r for r in rows if r["policy_category"] == "unseen_combination"]),
            "unseen_values_accuracy": accuracy([r for r in rows if r["policy_category"] == "unseen_values"]),
            "affected_accuracy": accuracy(changed), "affected_rows": len(changed),
            "unaffected_accuracy": accuracy(unchanged), "unaffected_rows": len(unchanged),
            "collateral_error_given_base_correct": (
                1 - accuracy(old_correct_unchanged) if old_correct_unchanged else None),
            "collateral_denominator": len(old_correct_unchanged),
            "base_and_updated_both_correct": mean(both),
            "schema_valid_rate": mean(parse(lookup[r["id"]]["text"]) is not None for r in rows),
            "mean_input_tokens": mean(token_values) if all(x is not None for x in token_values) else None,
        }
    return metrics, scores


def summarize(work, simulated=False):
    work = Path(work)
    manifest = verify_data(work / "data")
    prompts = read_rows(work / "data/eval_prompts.jsonl")
    reference = read_rows(work / "data/eval_reference.jsonl")
    inputs, tables, scores = {}, {}, {}
    provenance = {"simulated": simulated, "config": manifest["config"]}
    if simulated:
        paths = {endpoint: work / "simulation" / (endpoint + ".jsonl") for endpoint in ("base", *ARMS)}
    else:
        state = json.loads((work / "campaign.json").read_text(encoding="utf-8"))
        if state["status"] != "completed":
            raise ValueError("campaign incomplete: no final comparison table")
        expected_stages = {"preflight", *("train_" + arm for arm in ARMS),
                           *("infer_" + arm for arm in ("base", *ARMS))}
        if set(state["stages"]) != expected_stages:
            raise ValueError("complete training and evaluation inventory required")
        if state["identity"]["data_sha256"] != file_hash(work / "data/manifest.json"):
            raise ValueError("campaign data identity changed")
        paths = {}
        for key, entry in state["stages"].items():
            if entry["status"] != "completed":
                raise ValueError("campaign contains incomplete stages")
            stage = work / entry["directory"]
            job = json.loads((stage / "job.json").read_text(encoding="utf-8"))
            verify_stage(stage, job)
            inputs[key] = file_hash(stage / "receipt.json")
            if inputs[key] != entry["receipt_sha256"]:
                raise ValueError("completed stage receipt changed")
            if key == "preflight":
                preflight = json.loads((stage / "receipt.json").read_text(encoding="utf-8"))
                provenance.update(source_sha256=state["identity"]["source_sha256"],
                                  model_tree_sha256=preflight["model_inventory"]["tree_sha256"],
                                  packages=preflight["packages"],
                                  training_token_lengths=preflight.get("training_token_lengths"))
            if job["kind"] == "infer":
                paths[job["arm"]] = stage / "predictions.jsonl"
        if set(paths) != {"base", *ARMS}:
            raise ValueError("all five endpoints required")
    for endpoint, path in paths.items():
        predictions = read_rows(path)
        verify_predictions(predictions, prompts, simulated)
        tables[endpoint], scores[endpoint] = score_endpoint(reference, predictions)
        inputs[endpoint] = file_hash(path)
    config = manifest["config"]
    contrasts = {control: paired_interval(reference, scores["component_mix"], scores[control],
                                           seed=config["seed"], resamples=config["bootstrap_resamples"])
                 for control in ("whole_mix", "short_only", "full_only")}
    report = {
        "status": "simulation_only_no_model_evidence" if simulated else "exploratory_model_results",
        "claim_id": config["claim_id"], "claim_promotion": False, "seed": config["seed"],
        "data_manifest_sha256": file_hash(work / "data/manifest.json"), "input_sha256": inputs,
        "provenance": provenance,
        "metrics": tables, "component_mix_minus_control_short_unseen_values": contrasts,
        "oracle_structured_facts_accuracy": 1.0,
        "limitations": ["Synthetic structured facts, not natural-language extraction or a real business dataset.",
                       "Rule engine is exact and preferable when facts are already structured; no deployment superiority claimed.",
                       "One training seed and four fixed evaluation policies; intervals condition on this seed and these policies.",
                       "Module presence count matched between mix arms, not identical rendered tokens or FLOPs.",
                       "No legacy result is replaced. No chronology superiority, zero-prompt or generalization guarantee."],
    }
    out = work / ("simulation_summary" if simulated else "summary")
    out.mkdir(exist_ok=True)
    write_json(out / "summary.json", report)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["endpoint", "view", "accuracy", "unseen_values_accuracy", "affected_accuracy",
                         "unaffected_accuracy", "mean_input_tokens"])
        for endpoint, views in tables.items():
            for view, metrics in views.items():
                writer.writerow([endpoint, view, *[metrics[key] for key in (
                    "accuracy", "unseen_values_accuracy", "affected_accuracy", "unaffected_accuracy", "mean_input_tokens")]])
    lines = ["# SIMULATION — NOT MODEL EVIDENCE" if simulated else "# 单种子探索结果（不自动提升论文主张）",
             "", "模拟数据任务，结构化事实直接提供；规则引擎本身可精确求解。", "",
             "| 模型/策略 | Full 准确率 | Short 准确率 | Short 新取值准确率 |",
             "| --- | ---: | ---: | ---: |"]
    for endpoint, views in tables.items():
        lines.append(f"| {endpoint} | {views['full']['accuracy']:.3f} | {views['short']['accuracy']:.3f} | "
                     f"{views['short']['unseen_values_accuracy']:.3f} |")
    lines += ["", "## Component-mix 对照差值", "",
              "主观察：Short 的未见参数取值准确率差；区间按案例成组重采样，不代表跨种子稳定性。", ""]
    for control, values in contrasts.items():
        lines.append(f"- 相对 {control}: {values['mean_pp']:.2f} pp，描述性 95% 区间 {values['ci95_pp']}。")
    lines += ["", "不根据这张表自动追加训练、挑最佳 checkpoint 或宣称方法胜出。",
              "需要独立种子、规则版本和自然语言任务验证后才能作更强结论。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
