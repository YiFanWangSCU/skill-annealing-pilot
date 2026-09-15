"""Layered evaluator for ABCD next-action predictions."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import EXPOSURES


COMMON_PREFIX_ACTIONS = {"pull-up-account", "validate-purchase"}
DEFAULT_BOOTSTRAP_METRIC = "macro_tail_target_action_accuracy"
CASE_IDENTITY_FIELDS = (
    "task_id",
    "split",
    "case_id",
    "conversation_id",
    "conversation_hash",
    "source_turn_index",
    "flow",
    "subflow",
    "dialogue_prefix_hash",
    "target_output",
    "target_hash",
    "policy_covered",
    "common_contract_sha256",
    "user_prompt_sha256",
)
SAME_EXPOSURE_IDENTITY_FIELDS = CASE_IDENTITY_FIELDS + (
    "prompt_exposure",
    "policy_sha256",
    "messages",
)


def score_record(
    record: Mapping[str, Any],
    action_vocabulary: Sequence[str],
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    parsed, json_valid = parse_prediction(record)
    exact_keys = bool(json_valid and set(parsed) == {"action"})
    predicted_action = parsed.get("action") if json_valid else None
    action_type_valid = isinstance(predicted_action, str)
    action_known = bool(
        action_type_valid and predicted_action in set(action_vocabulary)
    )
    schema_valid = bool(exact_keys and action_known)
    gold_action = str(record["target_output"]["action"])
    action_correct = bool(action_type_valid and predicted_action == gold_action)
    subflow_actions = set((policy_actions or {}).get(str(record.get("subflow")), ()))
    illegal_policy_action = bool(
        action_known and subflow_actions and predicted_action not in subflow_actions
    )
    return {
        "case_id": record.get("case_id"),
        "conversation_id": record.get("conversation_id"),
        "subflow": record.get("subflow"),
        "prompt_exposure": record.get("prompt_exposure"),
        "target_action": gold_action,
        "predicted_action": predicted_action,
        "json_valid": float(json_valid),
        "schema_valid": float(schema_valid),
        "action_known": float(action_known),
        "action_correct": float(action_correct),
        "strict_action_correct": float(schema_valid and action_correct),
        "illegal_action": float(action_type_valid and not action_known),
        "illegal_policy_action": float(illegal_policy_action),
        "is_tail_action": gold_action not in COMMON_PREFIX_ACTIONS,
        "policy_covered": bool(record.get("policy_covered", True)),
    }


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    scores = [
        score_record(record, action_vocabulary, policy_actions) for record in records
    ]
    return aggregate_scores(scores)


def aggregate_scores(scores: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {
            "count": 0,
            "conversation_count": 0,
            "json_valid_rate": 0.0,
            "schema_valid_rate": 0.0,
            "action_accuracy": 0.0,
            "strict_action_accuracy": 0.0,
            "illegal_action_rate": 0.0,
            "illegal_policy_action_rate": 0.0,
            "macro_action_f1": 0.0,
            "macro_tail_action_f1": 0.0,
            "macro_subflow_action_accuracy": 0.0,
            "macro_target_action_accuracy": 0.0,
            "tail_action_accuracy": 0.0,
            "macro_tail_target_action_accuracy": 0.0,
        }

    def mean(rows: list[Mapping[str, Any]], key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    by_subflow: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for score in scores:
        by_subflow[str(score["subflow"])].append(score)
        by_action[str(score["target_action"])].append(score)
    tail = [score for score in scores if score["is_tail_action"]]
    per_subflow = {
        key: mean(rows, "action_correct") for key, rows in sorted(by_subflow.items())
    }
    per_action = {
        key: mean(rows, "action_correct") for key, rows in sorted(by_action.items())
    }
    tail_per_action = {
        action: accuracy
        for action, accuracy in per_action.items()
        if action not in COMMON_PREFIX_ACTIONS
    }
    action_f1 = _per_class_f1(scores)
    tail_action_f1 = {
        action: value
        for action, value in action_f1.items()
        if action not in COMMON_PREFIX_ACTIONS
    }
    return {
        "count": len(scores),
        "conversation_count": len({str(score["conversation_id"]) for score in scores}),
        "json_valid_rate": mean(scores, "json_valid"),
        "schema_valid_rate": mean(scores, "schema_valid"),
        "action_accuracy": mean(scores, "action_correct"),
        "strict_action_accuracy": mean(scores, "strict_action_correct"),
        "illegal_action_rate": mean(scores, "illegal_action"),
        "illegal_policy_action_rate": mean(scores, "illegal_policy_action"),
        "macro_action_f1": sum(action_f1.values()) / len(action_f1),
        "macro_tail_action_f1": (
            sum(tail_action_f1.values()) / len(tail_action_f1)
            if tail_action_f1
            else 0.0
        ),
        "macro_subflow_action_accuracy": sum(per_subflow.values()) / len(per_subflow),
        "macro_target_action_accuracy": sum(per_action.values()) / len(per_action),
        "tail_action_count": len(tail),
        "tail_action_accuracy": mean(tail, "action_correct"),
        "macro_tail_target_action_accuracy": (
            sum(tail_per_action.values()) / len(tail_per_action)
            if tail_per_action
            else 0.0
        ),
        "per_subflow_action_accuracy": per_subflow,
        "per_target_action_accuracy": per_action,
        "per_target_action_f1": action_f1,
    }


def evaluate_by_exposure(
    records: Iterable[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    groups = _records_by_exposure(list(records))
    _validate_complete_exposure_panel(groups)
    output = {
        exposure: evaluate_records(groups[exposure], action_vocabulary, policy_actions)
        for exposure in EXPOSURES
    }
    if "full" in output and "no_policy" in output:
        full = output["full"]["macro_tail_target_action_accuracy"]
        no_policy = output["no_policy"]["macro_tail_target_action_accuracy"]
        output["retention"] = {
            "full_to_no_policy_macro_tail_gap": full - no_policy,
            "no_policy_over_full": no_policy / full if full else 0.0,
            "worst_exposure_macro_tail_target_accuracy": min(
                metrics["macro_tail_target_action_accuracy"]
                for exposure, metrics in output.items()
                if exposure != "retention"
            ),
        }
    return output


def _per_class_f1(scores: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    classes = sorted({str(score["target_action"]) for score in scores})
    output: dict[str, float] = {}
    for action in classes:
        true_positive = sum(
            score["target_action"] == action and score.get("predicted_action") == action
            for score in scores
        )
        false_positive = sum(
            score["target_action"] != action and score.get("predicted_action") == action
            for score in scores
        )
        false_negative = sum(
            score["target_action"] == action and score.get("predicted_action") != action
            for score in scores
        )
        denominator = 2 * true_positive + false_positive + false_negative
        output[action] = 2 * true_positive / denominator if denominator else 0.0
    return output


def cluster_bootstrap_interval(
    records: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    *,
    metric: str = DEFAULT_BOOTSTRAP_METRIC,
    samples: int = 2000,
    seed: int = 20260827,
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("cluster bootstrap samples must be positive")
    by_conversation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_conversation[str(record["conversation_id"])].append(record)
    clusters = sorted(by_conversation)
    if not clusters:
        raise ValueError("cluster bootstrap requires at least one conversation")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [generator.choice(clusters) for _ in clusters]
        rows = [row for cluster in sampled for row in by_conversation[cluster]]
        estimates.append(
            float(evaluate_records(rows, action_vocabulary, policy_actions)[metric])
        )
    estimates.sort()
    return {
        "metric": metric,
        "cluster_key": "conversation_id",
        "cluster_count": len(clusters),
        "samples": samples,
        "seed": seed,
        "estimate": float(
            evaluate_records(records, action_vocabulary, policy_actions)[metric]
        ),
        "ci_95_low": estimates[round((samples - 1) * 0.025)],
        "ci_95_high": estimates[round((samples - 1) * 0.975)],
    }


def paired_cluster_bootstrap_difference(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    *,
    metric: str = DEFAULT_BOOTSTRAP_METRIC,
    samples: int = 2000,
    seed: int = 20260827,
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("paired cluster bootstrap samples must be positive")
    candidate_by_key = _unique_rows_by_key(candidate, "pairing_key")
    reference_by_key = _unique_rows_by_key(reference, "pairing_key")
    if set(candidate_by_key) != set(reference_by_key):
        raise ValueError("paired prediction files do not share the same pairing keys")
    by_conversation: dict[str, list[str]] = defaultdict(list)
    for key, row in candidate_by_key.items():
        reference_row = reference_by_key[key]
        _require_matching_fields(
            row,
            reference_row,
            SAME_EXPOSURE_IDENTITY_FIELDS,
            context="paired rows",
        )
        by_conversation[str(row["conversation_id"])].append(key)
    clusters = sorted(by_conversation)
    if not clusters:
        raise ValueError("paired bootstrap requires at least one conversation")
    generator = random.Random(seed)
    differences = []
    for _ in range(samples):
        sampled = [generator.choice(clusters) for _ in clusters]
        keys = [key for cluster in sampled for key in by_conversation[cluster]]
        candidate_rows = [candidate_by_key[key] for key in keys]
        reference_rows = [reference_by_key[key] for key in keys]
        candidate_metric = evaluate_records(
            candidate_rows, action_vocabulary, policy_actions
        )[metric]
        reference_metric = evaluate_records(
            reference_rows, action_vocabulary, policy_actions
        )[metric]
        differences.append(float(candidate_metric) - float(reference_metric))
    differences.sort()
    candidate_metric = evaluate_records(candidate, action_vocabulary, policy_actions)[
        metric
    ]
    reference_metric = evaluate_records(reference, action_vocabulary, policy_actions)[
        metric
    ]
    return {
        "metric": metric,
        "cluster_key": "conversation_id",
        "cluster_count": len(clusters),
        "samples": samples,
        "seed": seed,
        "candidate_estimate": float(candidate_metric),
        "reference_estimate": float(reference_metric),
        "candidate_minus_reference": float(candidate_metric) - float(reference_metric),
        "ci_95_low": differences[round((samples - 1) * 0.025)],
        "ci_95_high": differences[round((samples - 1) * 0.975)],
    }


def cluster_bootstrap_by_exposure(
    records: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    *,
    metric: str = DEFAULT_BOOTSTRAP_METRIC,
    samples: int = 2000,
    seed: int = 20260827,
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Bootstrap each exposure independently and the paired retention gap.

    The point metrics are defined per exposure, so pooling all four prompt
    renderings into one bootstrap estimate is not a valid uncertainty estimate
    for the pre-registered No-Policy endpoint.
    """

    groups = _records_by_exposure(records)
    _validate_complete_exposure_panel(groups)
    by_exposure = {
        exposure: cluster_bootstrap_interval(
            groups[exposure],
            action_vocabulary,
            metric=metric,
            samples=samples,
            seed=seed,
            policy_actions=policy_actions,
        )
        for exposure in EXPOSURES
        if exposure in groups
    }
    output: dict[str, Any] = {
        "metric": metric,
        "resampling_unit": "conversation_id_within_exposure",
        "by_exposure": by_exposure,
    }
    if "full" in groups and "no_policy" in groups:
        output["retention"] = paired_exposure_cluster_bootstrap_gap(
            records,
            action_vocabulary,
            metric=metric,
            samples=samples,
            seed=seed,
            policy_actions=policy_actions,
        )
    return output


def paired_cluster_bootstrap_difference_by_exposure(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    *,
    metric: str = DEFAULT_BOOTSTRAP_METRIC,
    samples: int = 2000,
    seed: int = 20260827,
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Return matched candidate-reference intervals for every exposure."""

    candidate_groups = _records_by_exposure(candidate)
    reference_groups = _records_by_exposure(reference)
    _validate_complete_exposure_panel(candidate_groups)
    _validate_complete_exposure_panel(reference_groups)
    if set(candidate_groups) != set(reference_groups):
        raise ValueError("candidate and reference expose different prompt conditions")
    return {
        "metric": metric,
        "contrast": "candidate_minus_reference",
        "resampling_unit": "matched_conversation_id_within_exposure",
        "by_exposure": {
            exposure: paired_cluster_bootstrap_difference(
                candidate_groups[exposure],
                reference_groups[exposure],
                action_vocabulary,
                metric=metric,
                samples=samples,
                seed=seed,
                policy_actions=policy_actions,
            )
            for exposure in EXPOSURES
            if exposure in candidate_groups
        },
    }


def paired_exposure_cluster_bootstrap_gap(
    records: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    *,
    metric: str = DEFAULT_BOOTSTRAP_METRIC,
    samples: int = 2000,
    seed: int = 20260827,
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Estimate Full minus No-Policy using matched conversation resamples."""

    if samples <= 0:
        raise ValueError("retention bootstrap samples must be positive")
    full_by_case, no_policy_by_case = _matched_exposure_case_maps(records)
    by_conversation = _paired_cases_by_conversation(full_by_case, no_policy_by_case)
    clusters = sorted(by_conversation)
    generator = random.Random(seed)
    gaps: list[float] = []
    for _ in range(samples):
        sampled = [generator.choice(clusters) for _ in clusters]
        case_ids = [
            case_id
            for conversation_id in sampled
            for case_id in by_conversation[conversation_id]
        ]
        full_metric = evaluate_records(
            [full_by_case[case_id] for case_id in case_ids],
            action_vocabulary,
            policy_actions,
        )[metric]
        no_policy_metric = evaluate_records(
            [no_policy_by_case[case_id] for case_id in case_ids],
            action_vocabulary,
            policy_actions,
        )[metric]
        gaps.append(float(full_metric) - float(no_policy_metric))
    gaps.sort()
    full_estimate = float(
        evaluate_records(
            list(full_by_case.values()), action_vocabulary, policy_actions
        )[metric]
    )
    no_policy_estimate = float(
        evaluate_records(
            list(no_policy_by_case.values()), action_vocabulary, policy_actions
        )[metric]
    )
    return {
        "metric": metric,
        "contrast": "full_minus_no_policy",
        "formula": "Full - No-Policy",
        "cluster_key": "conversation_id",
        "pair_key": "case_id",
        "cluster_count": len(clusters),
        "paired_case_count": len(full_by_case),
        "samples": samples,
        "seed": seed,
        "full_estimate": full_estimate,
        "no_policy_estimate": no_policy_estimate,
        "full_minus_no_policy": full_estimate - no_policy_estimate,
        "ci_95_low": gaps[round((samples - 1) * 0.025)],
        "ci_95_high": gaps[round((samples - 1) * 0.975)],
    }


def paired_cluster_bootstrap_gap_of_gaps(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    action_vocabulary: Sequence[str],
    *,
    metric: str = DEFAULT_BOOTSTRAP_METRIC,
    samples: int = 2000,
    seed: int = 20260827,
    policy_actions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compare two arms' matched Full-to-No-Policy retention gaps."""

    if samples <= 0:
        raise ValueError("gap-of-gaps bootstrap samples must be positive")
    candidate_full, candidate_no_policy = _matched_exposure_case_maps(candidate)
    reference_full, reference_no_policy = _matched_exposure_case_maps(reference)
    expected_cases = set(candidate_full)
    if (
        set(reference_full) != expected_cases
        or set(candidate_no_policy) != expected_cases
        or set(reference_no_policy) != expected_cases
    ):
        raise ValueError("candidate and reference do not share matched exposure cases")
    for case_id in sorted(expected_cases):
        for exposure, candidate_rows, reference_rows in (
            ("full", candidate_full, reference_full),
            ("no_policy", candidate_no_policy, reference_no_policy),
        ):
            _require_matching_fields(
                candidate_rows[case_id],
                reference_rows[case_id],
                SAME_EXPOSURE_IDENTITY_FIELDS,
                context=f"candidate/reference {exposure} rows",
            )
    by_conversation = _paired_cases_by_conversation(candidate_full, candidate_no_policy)
    clusters = sorted(by_conversation)
    generator = random.Random(seed)
    gap_differences: list[float] = []
    gain_differences: list[float] = []
    for _ in range(samples):
        sampled = [generator.choice(clusters) for _ in clusters]
        case_ids = [
            case_id
            for conversation_id in sampled
            for case_id in by_conversation[conversation_id]
        ]

        def sampled_metric(rows: Mapping[str, Mapping[str, Any]]) -> float:
            return float(
                evaluate_records(
                    [rows[case_id] for case_id in case_ids],
                    action_vocabulary,
                    policy_actions,
                )[metric]
            )

        candidate_full_metric = sampled_metric(candidate_full)
        candidate_no_policy_metric = sampled_metric(candidate_no_policy)
        reference_full_metric = sampled_metric(reference_full)
        reference_no_policy_metric = sampled_metric(reference_no_policy)
        gap_difference = (
            candidate_full_metric
            - candidate_no_policy_metric
            - reference_full_metric
            + reference_no_policy_metric
        )
        gap_differences.append(gap_difference)
        gain_differences.append(-gap_difference)

    def point_metric(rows: Mapping[str, Mapping[str, Any]]) -> float:
        return float(
            evaluate_records(list(rows.values()), action_vocabulary, policy_actions)[
                metric
            ]
        )

    candidate_full_metric = point_metric(candidate_full)
    candidate_no_policy_metric = point_metric(candidate_no_policy)
    reference_full_metric = point_metric(reference_full)
    reference_no_policy_metric = point_metric(reference_no_policy)
    candidate_gap = candidate_full_metric - candidate_no_policy_metric
    reference_gap = reference_full_metric - reference_no_policy_metric
    gap_difference = candidate_gap - reference_gap
    gap_differences.sort()
    gain_differences.sort()
    low_index = round((samples - 1) * 0.025)
    high_index = round((samples - 1) * 0.975)
    return {
        "metric": metric,
        "cluster_key": "conversation_id",
        "pair_key": "case_id",
        "cluster_count": len(clusters),
        "paired_case_count": len(expected_cases),
        "samples": samples,
        "seed": seed,
        "candidate_full_estimate": candidate_full_metric,
        "candidate_no_policy_estimate": candidate_no_policy_metric,
        "reference_full_estimate": reference_full_metric,
        "reference_no_policy_estimate": reference_no_policy_metric,
        "candidate_full_minus_no_policy": candidate_gap,
        "reference_full_minus_no_policy": reference_gap,
        "candidate_minus_reference_gap": {
            "formula": (
                "(Candidate Full - Candidate No-Policy) - "
                "(Reference Full - Reference No-Policy)"
            ),
            "estimate": gap_difference,
            "ci_95_low": gap_differences[low_index],
            "ci_95_high": gap_differences[high_index],
        },
        "no_policy_gain_minus_full_gain": {
            "formula": (
                "(Candidate No-Policy - Reference No-Policy) - "
                "(Candidate Full - Reference Full)"
            ),
            "estimate": -gap_difference,
            "ci_95_low": gain_differences[low_index],
            "ci_95_high": gain_differences[high_index],
        },
    }


def _records_by_exposure(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        exposure = str(record.get("prompt_exposure", ""))
        if exposure not in EXPOSURES:
            raise ValueError(f"unknown or missing prompt exposure: {exposure!r}")
        groups[exposure].append(record)
    if not groups:
        raise ValueError("exposure analysis requires at least one record")
    return groups


def _unique_rows_by_key(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if row.get(key) is None:
            raise ValueError(f"paired analysis requires {key}")
        value = str(row[key])
        if value in output:
            raise ValueError(f"duplicate {key}: {value}")
        output[value] = row
    return output


def _matched_exposure_case_maps(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    groups = _records_by_exposure(records)
    by_exposure = _validate_complete_exposure_panel(groups)
    return by_exposure["full"], by_exposure["no_policy"]


def _validate_complete_exposure_panel(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    missing = [exposure for exposure in EXPOSURES if exposure not in groups]
    if missing:
        raise ValueError(f"complete exposure panel is missing: {missing}")
    by_exposure = {
        exposure: _unique_rows_by_key(groups[exposure], "case_id")
        for exposure in EXPOSURES
    }
    expected_cases = set(by_exposure[EXPOSURES[0]])
    for exposure in EXPOSURES[1:]:
        if set(by_exposure[exposure]) != expected_cases:
            raise ValueError("exposure panels do not share the same case IDs")
    for case_id in sorted(expected_cases):
        reference = by_exposure[EXPOSURES[0]][case_id]
        reference_user_messages = _user_messages(reference)
        for exposure in EXPOSURES[1:]:
            row = by_exposure[exposure][case_id]
            _require_matching_fields(
                reference,
                row,
                CASE_IDENTITY_FIELDS,
                context="matched exposures",
            )
            if _user_messages(row) != reference_user_messages:
                raise ValueError("matched exposures disagree on user messages")
    return by_exposure


def _require_matching_fields(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    fields: Sequence[str],
    *,
    context: str,
) -> None:
    for field in fields:
        if field not in first or field not in second:
            raise ValueError(f"{context} require identity field {field}")
        if first[field] != second[field]:
            raise ValueError(f"{context} disagree on {field}")


def _user_messages(row: Mapping[str, Any]) -> list[Any]:
    if "messages" not in row or not isinstance(row["messages"], list):
        raise ValueError("matched exposure analysis requires messages")
    return [
        message
        for message in row["messages"]
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]


def _paired_cases_by_conversation(
    first_by_case: Mapping[str, Mapping[str, Any]],
    second_by_case: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    if set(first_by_case) != set(second_by_case):
        raise ValueError("paired exposure maps do not share the same cases")
    output: dict[str, list[str]] = defaultdict(list)
    for case_id, first_row in first_by_case.items():
        conversation_id = str(first_row["conversation_id"])
        if conversation_id != str(second_by_case[case_id]["conversation_id"]):
            raise ValueError("paired exposure maps disagree on conversation identity")
        output[conversation_id].append(case_id)
    if not output:
        raise ValueError("paired bootstrap requires at least one conversation")
    for case_ids in output.values():
        case_ids.sort()
    return output


def parse_prediction(record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if record.get("oracle"):
        value = record.get("target_output")
    else:
        value = record.get(
            "predicted_output",
            record.get("prediction", record.get("response")),
        )
    if isinstance(value, Mapping):
        return dict(value), True
    if not isinstance(value, str):
        return {}, False
    text = _strip_code_fence(value.strip())
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed, True
    return {}, False


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _is_locked_test_path(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/").casefold()
    return ".locked." in normalized or "/test_" in normalized


def authorize_prediction_path(
    path: str | Path,
    *,
    allow_locked_test: bool,
    final_eval_manifest: str | Path | None,
) -> None:
    if not _is_locked_test_path(path):
        return
    if not allow_locked_test or final_eval_manifest is None:
        raise PermissionError(
            "locked ABCD test predictions require --allow-locked-test and "
            "--final-eval-manifest"
        )
    manifest = json.loads(Path(final_eval_manifest).read_text(encoding="utf-8"))
    if (
        manifest.get("test_unlock") is not True
        or manifest.get("model_selection_frozen") is not True
        or not isinstance(manifest.get("run_id"), str)
        or not manifest["run_id"].strip()
    ):
        raise PermissionError("final-eval manifest does not authorize test unlock")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ABCD action predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prompt-registry", required=True)
    parser.add_argument("--by-exposure", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--compare-predictions")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--final-eval-manifest")
    parser.add_argument("--output")
    args = parser.parse_args()

    authorize_prediction_path(
        args.predictions,
        allow_locked_test=args.allow_locked_test,
        final_eval_manifest=args.final_eval_manifest,
    )
    if args.compare_predictions:
        authorize_prediction_path(
            args.compare_predictions,
            allow_locked_test=args.allow_locked_test,
            final_eval_manifest=args.final_eval_manifest,
        )
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples cannot be negative")
    records = load_jsonl(args.predictions)
    if args.oracle:
        records = [{**record, "oracle": True} for record in records]
    registry = json.loads(Path(args.prompt_registry).read_text(encoding="utf-8"))
    vocabulary = registry["global_action_vocabulary"]
    policy_actions = registry.get("policy_actions", {})
    effective_bootstrap_samples = args.bootstrap_samples
    if args.compare_predictions and not effective_bootstrap_samples:
        effective_bootstrap_samples = 2000

    def arm_report(arm_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = (
            evaluate_by_exposure(arm_records, vocabulary, policy_actions)
            if args.by_exposure
            else evaluate_records(arm_records, vocabulary, policy_actions)
        )
        if not effective_bootstrap_samples:
            return metrics
        bootstrap = (
            cluster_bootstrap_by_exposure(
                arm_records,
                vocabulary,
                samples=effective_bootstrap_samples,
                seed=args.bootstrap_seed,
                policy_actions=policy_actions,
            )
            if args.by_exposure
            else cluster_bootstrap_interval(
                arm_records,
                vocabulary,
                samples=effective_bootstrap_samples,
                seed=args.bootstrap_seed,
                policy_actions=policy_actions,
            )
        )
        return {"metrics": metrics, "cluster_bootstrap": bootstrap}

    report = arm_report(records)
    if args.compare_predictions:
        reference = load_jsonl(args.compare_predictions)
        paired = (
            paired_cluster_bootstrap_difference_by_exposure(
                records,
                reference,
                vocabulary,
                samples=effective_bootstrap_samples,
                seed=args.bootstrap_seed,
                policy_actions=policy_actions,
            )
            if args.by_exposure
            else paired_cluster_bootstrap_difference(
                records,
                reference,
                vocabulary,
                samples=effective_bootstrap_samples,
                seed=args.bootstrap_seed,
                policy_actions=policy_actions,
            )
        )
        if args.by_exposure:
            paired["retention_gap_of_gaps"] = paired_cluster_bootstrap_gap_of_gaps(
                records,
                reference,
                vocabulary,
                samples=effective_bootstrap_samples,
                seed=args.bootstrap_seed,
                policy_actions=policy_actions,
            )
        report = {
            "candidate": report,
            "reference": arm_report(reference),
            "paired_cluster_bootstrap": paired,
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
