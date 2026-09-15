"""Non-neural workflow-position shortcut baselines for ABCD."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Hashable, Mapping, Sequence

from .evaluator import evaluate_records


def evaluate_shortcut_baselines(
    train_examples: Sequence[Mapping[str, Any]],
    eval_examples: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
) -> dict[str, Any]:
    global_action = _majority(
        str(row["target_output"]["action"]) for row in train_examples
    )
    specifications: dict[str, Callable[[Mapping[str, Any]], Hashable]] = {
        "subflow_majority": lambda row: str(row["subflow"]),
        "subflow_step": lambda row: (
            str(row["subflow"]),
            int(row["prior_action_count"]),
        ),
        "subflow_last_action": lambda row: (
            str(row["subflow"]),
            row.get("last_action"),
        ),
    }
    reports: dict[str, Any] = {}
    global_predictions = [
        {**row, "predicted_output": {"action": global_action}} for row in eval_examples
    ]
    reports["global_majority"] = {
        "fallback_count": 0,
        "metrics": evaluate_records(global_predictions, action_vocabulary),
    }
    for name, key_fn in specifications.items():
        table = _fit_lookup(train_examples, key_fn)
        fallback_count = 0
        predictions = []
        for row in eval_examples:
            action = table.get(key_fn(row))
            if action is None:
                fallback_count += 1
                action = global_action
            predictions.append({**row, "predicted_output": {"action": action}})
        reports[name] = {
            "lookup_size": len(table),
            "fallback_count": fallback_count,
            "metrics": evaluate_records(predictions, action_vocabulary),
        }
    step_accuracy = reports["subflow_step"]["metrics"]["action_accuracy"]
    strongest_overall_name, strongest_overall = max(
        reports.items(), key=lambda item: item[1]["metrics"]["action_accuracy"]
    )
    strongest_primary_name, strongest_primary = max(
        reports.items(),
        key=lambda item: item[1]["metrics"]["macro_tail_target_action_accuracy"],
    )
    strongest_overall_accuracy = strongest_overall["metrics"]["action_accuracy"]
    strongest_primary_accuracy = strongest_primary["metrics"][
        "macro_tail_target_action_accuracy"
    ]
    return {
        "fit_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "global_majority_action": global_action,
        "baselines": reports,
        "subflow_step_action_accuracy": step_accuracy,
        "strongest_overall_shortcut": strongest_overall_name,
        "strongest_overall_shortcut_action_accuracy": strongest_overall_accuracy,
        "strongest_primary_shortcut": strongest_primary_name,
        "strongest_primary_shortcut_macro_tail_accuracy": strongest_primary_accuracy,
        "overall_action_metric_decisive": strongest_overall_accuracy < 0.90,
        "primary_tail_metric_decisive": strongest_primary_accuracy < 0.90,
        "action_only_saturation_threshold": 0.90,
    }


def _fit_lookup(
    examples: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Hashable],
) -> dict[Hashable, str]:
    counts: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for row in examples:
        counts[key_fn(row)][str(row["target_output"]["action"])] += 1
    return {key: _majority(counter.elements()) for key, counter in counts.items()}


def _majority(values: Any) -> str:
    counter = Counter(values)
    if not counter:
        raise ValueError("cannot compute a majority label from an empty collection")
    max_count = max(counter.values())
    return sorted(key for key, count in counter.items() if count == max_count)[0]
