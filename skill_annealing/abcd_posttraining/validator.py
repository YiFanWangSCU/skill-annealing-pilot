"""Fail-closed static gates for the ABCD post-training bundle."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import (
    EXPECTED_PRODUCT_DEFECT_INVENTORY,
    EXPOSURES,
    PRODUCT_DEFECT_SUBFLOWS,
)
from .dataset import (
    build_policy_action_sets,
    canonical_json,
    extract_action_examples,
    flatten_action_vocabulary,
    load_abcd,
    load_json,
    normalize_values,
    sha256_text,
)
from .evaluator import evaluate_records
from .prompts import (
    build_messages,
    build_prompt_registry,
    prompt_component_hashes,
)
from .source import sha256_file, verify_sources


def validate_bundle(
    data_dir: str | Path,
    *,
    raw_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(data_dir)
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    manifest = _load_json(root / "build_manifest.json")
    inventory = _load_json(root / "inventory.json")
    registry = _load_json(root / "prompt_registry.json")
    shortcuts = _load_json(root / "shortcut_baselines_dev.json")
    vocabulary = tuple(registry["global_action_vocabulary"])
    _validate_artifact_manifest(root, manifest, errors)
    reconstructed_examples: dict[str, list[dict[str, Any]]] | None = None
    reconstructed_inventory: dict[str, Any] | None = None

    if manifest.get("no_routing") is not True:
        errors.append("manifest must explicitly set no_routing=true")
    if manifest.get("oracle_workflow_fields") != ["flow", "subflow"]:
        errors.append("oracle workflow fields must be exactly flow/subflow")
    if manifest.get("target_schema") != {"action": "official_action_code"}:
        errors.append("target schema must be action-only")
    if manifest.get("raw_values_used_as_target") is not False:
        errors.append("raw ABCD values must not be model targets")
    if manifest.get("test_lock", {}).get("locked") is not True:
        errors.append("test panel is not locked")

    if manifest.get("official_source"):
        if raw_dir is None:
            errors.append("raw_dir is required to verify an official build")
        else:
            try:
                verify_sources(raw_dir)
                raw = Path(raw_dir)
                source_dataset = load_abcd(raw / "abcd_v1.1.json.gz")
                source_guidelines = load_json(raw / "guidelines.json")
                source_ontology = load_json(raw / "ontology.json")
                source_vocabulary = flatten_action_vocabulary(source_ontology)
                source_policy_actions = build_policy_action_sets(
                    source_guidelines, source_vocabulary
                )
                reconstructed_examples, reconstructed_inventory = (
                    extract_action_examples(
                        source_dataset,
                        policy_actions=source_policy_actions,
                        action_vocabulary=source_vocabulary,
                    )
                )
                expected_registry = build_prompt_registry(
                    source_guidelines, source_vocabulary
                )
                if canonical_json(registry) != canonical_json(expected_registry):
                    errors.append(
                        "prompt registry differs from the frozen guideline/ontology source"
                    )
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"source verification failed: {exc}")
        _validate_official_inventory(inventory, errors)

    if float(inventory.get("mapping_coverage", 0.0)) < 0.98:
        errors.append("action target mapping coverage is below 98%")
    for split in ("train", "dev", "test"):
        reported = inventory.get("splits", {}).get(split, {})
        missing = [
            subflow
            for subflow in PRODUCT_DEFECT_SUBFLOWS
            if reported.get("subflow_conversations", {}).get(subflow, 0) == 0
        ]
        if missing:
            errors.append(f"{split} is missing subflows: {missing}")

    all_examples = {
        split: _load_jsonl(root / "examples" / f"{split}_all_actions.jsonl")
        for split in ("train", "dev", "test")
    }
    main_examples = {
        split: _load_jsonl(root / "examples" / f"{split}_policy_clean.jsonl")
        for split in ("train", "dev", "test")
    }
    if reconstructed_examples is not None:
        for split in ("train", "dev", "test"):
            if [canonical_json(row) for row in all_examples[split]] != [
                canonical_json(row) for row in reconstructed_examples[split]
            ]:
                errors.append(
                    f"{split}: generated examples differ from frozen source reconstruction"
                )
        assert reconstructed_inventory is not None
        reconstructed_inventory["policy_action_sets"] = {
            subflow: sorted(actions)
            for subflow, actions in source_policy_actions.items()
        }
        reconstructed_inventory["main_view"] = "conversation_policy_clean"
        reconstructed_inventory["main_split_counts"] = {
            split: sum(
                bool(row["conversation_policy_clean"])
                for row in reconstructed_examples[split]
            )
            for split in ("train", "dev", "test")
        }
        reconstructed_inventory["all_action_split_counts"] = {
            split: len(reconstructed_examples[split])
            for split in ("train", "dev", "test")
        }
        if canonical_json(inventory) != canonical_json(reconstructed_inventory):
            errors.append("inventory differs from frozen source reconstruction")
    _validate_examples(all_examples, main_examples, vocabulary, errors)
    _validate_generated_counts(all_examples, main_examples, inventory, manifest, errors)
    _validate_split_isolation(all_examples, errors)

    staged = _load_jsonl(root / "arms" / "train_staged_order.jsonl")
    random_mix = _load_jsonl(root / "arms" / "train_random_mix.jsonl")
    full_only = _load_jsonl(root / "arms" / "train_full_only.jsonl")
    no_policy_only = _load_jsonl(root / "arms" / "train_no_policy_only.jsonl")
    arm_rows = {
        "train_staged_order": staged,
        "train_random_mix": random_mix,
        "train_full_only": full_only,
        "train_no_policy_only": no_policy_only,
    }
    _validate_training_records(
        root,
        arm_rows,
        main_examples["train"],
        registry,
        int(manifest.get("order_seed", -1)),
        errors,
    )
    staged_ids = [row["record_id"] for row in staged]
    random_ids = [row["record_id"] for row in random_mix]
    if Counter(staged_ids) != Counter(random_ids):
        errors.append("Random-Mix and Staged record multisets differ")
    if staged_ids == random_ids:
        errors.append("Random-Mix and Staged orders are identical")
    if len(full_only) != len(staged) or len(no_policy_only) != len(staged):
        errors.append("single-exposure arms do not match mixed-arm record counts")
    if Counter(row["prompt_exposure"] for row in staged) != Counter(
        {exposure: len(main_examples["train"]) for exposure in EXPOSURES}
    ):
        errors.append("Staged exposure counts are not exactly balanced")
    if [row["prompt_exposure"] for row in staged] != sorted(
        [row["prompt_exposure"] for row in staged],
        key=EXPOSURES.index,
    ):
        errors.append("Staged arm is not ordered Full->Partial->Minimal->No-Policy")
    for arm_name, rows in arm_rows.items():
        hashes = manifest.get("arm_hashes", {}).get(arm_name, {})
        ordered_hash = sha256_text(canonical_json([row["record_id"] for row in rows]))
        multiset_hash = sha256_text(
            canonical_json(sorted(row["record_id"] for row in rows))
        )
        if hashes.get("ordered_sha256") != ordered_hash:
            errors.append(f"{arm_name}: manifest ordered hash mismatch")
        if hashes.get("record_multiset_sha256") != multiset_hash:
            errors.append(f"{arm_name}: manifest multiset hash mismatch")

    dev_eval = _load_jsonl(root / "eval" / "dev_policy_clean_eval.jsonl")
    test_eval = _load_jsonl(root / "eval" / "test_policy_clean_eval.locked.jsonl")
    _validate_eval_panel(dev_eval, main_examples["dev"], registry, errors, "dev")
    _validate_eval_panel(test_eval, main_examples["test"], registry, errors, "test")
    oracle_rows = [{**row, "oracle": True} for row in dev_eval]
    oracle = evaluate_records(oracle_rows, vocabulary)
    if oracle["action_accuracy"] != 1.0 or oracle["strict_action_accuracy"] != 1.0:
        errors.append("oracle evaluator did not reach 100% strict action accuracy")

    step_accuracy = float(shortcuts.get("subflow_step_action_accuracy", 0.0))
    strongest_overall = float(
        shortcuts.get("strongest_overall_shortcut_action_accuracy", step_accuracy)
    )
    strongest_primary = float(
        shortcuts.get("strongest_primary_shortcut_macro_tail_accuracy", step_accuracy)
    )
    threshold = float(shortcuts.get("action_only_saturation_threshold", 0.90))
    if strongest_primary >= threshold:
        blockers.append(
            "a workflow-position shortcut saturates the macro tail-action primary "
            "metric; strong policy-internalization claim is blocked"
        )
    if strongest_overall >= threshold:
        warnings.append(
            "overall action accuracy is shortcut-saturated; report tail-action and "
            "macro-by-action metrics as primary"
        )
    low_cluster_cells = _low_conversation_cluster_cells(main_examples)
    if low_cluster_cells:
        warnings.append(
            "some policy-clean dev/test subflows contain fewer than 20 "
            f"conversation clusters: {low_cluster_cells}"
        )
    if float(inventory["splits"]["train"].get("policy_covered_rate", 0.0)) < 0.90:
        blockers.append("less than 90% of Product Defect actions are guideline-covered")

    return {
        "local_static_valid": not errors,
        "gpu_smoke_ready": not errors,
        "formal_training_ready": not errors and not blockers,
        "static_claim_prerequisites_ready": not errors and not blockers,
        "formal_claim_ready": False,
        "formal_claim_pending_requirements": [
            "matched model runs are not part of this static validator",
            "paired conversation-cluster confidence intervals are still required",
            "the locked test may be opened only after model selection is frozen",
        ],
        "gpu_training_recommended": not errors and not blockers,
        "error_count": len(errors),
        "errors": errors,
        "claim_blocker_count": len(blockers),
        "claim_blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "oracle_dev_policy_clean": oracle,
        "shortcut_subflow_step_action_accuracy": step_accuracy,
        "strongest_overall_shortcut_action_accuracy": strongest_overall,
        "strongest_primary_shortcut_macro_tail_accuracy": strongest_primary,
        "overall_action_metric_decisive": strongest_overall < threshold,
        "primary_tail_metric_decisive": strongest_primary < threshold,
        "main_split_counts": {
            split: len(rows) for split, rows in main_examples.items()
        },
        "all_action_split_counts": {
            split: len(rows) for split, rows in all_examples.items()
        },
        "low_conversation_cluster_cells": low_cluster_cells,
    }


def reject_locked_test_path(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    if ".locked." in normalized or "/test_" in normalized:
        raise PermissionError(
            "locked ABCD test data cannot be used for training or selection"
        )


def _validate_examples(
    all_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    main_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    vocabulary: Sequence[str],
    errors: list[str],
) -> None:
    known = set(vocabulary)
    for split, rows in all_examples.items():
        case_ids = set()
        for row in rows:
            case_id = str(row.get("case_id"))
            if case_id in case_ids:
                errors.append(f"duplicate case_id in {split}: {case_id}")
            case_ids.add(case_id)
            if set(row.get("target_output", {})) != {"action"}:
                errors.append(f"{case_id}: target is not action-only")
            if row.get("target_output", {}).get("action") not in known:
                errors.append(f"{case_id}: target action is outside ontology")
            expected_case_id = (
                f"abcd_{split}_{row.get('conversation_id')}_i"
                f"{row.get('source_turn_index')}"
            )
            if case_id != expected_case_id:
                errors.append(f"{case_id}: case identity is not array-index based")
            if "scenario" in row:
                errors.append(f"{case_id}: hidden scenario leaked into example")
            raw_values = row.get("raw_target_values")
            if not isinstance(raw_values, list) or not all(
                isinstance(value, str) for value in raw_values
            ):
                errors.append(
                    f"{case_id}: raw target values are not an ordered string list"
                )
            elif row.get("normalized_target_values") != normalize_values(raw_values):
                errors.append(f"{case_id}: normalized target values are inconsistent")
            prefix = row.get("dialogue_prefix")
            if not isinstance(prefix, list):
                errors.append(f"{case_id}: dialogue prefix is not a list")
                continue
            if len(prefix) != int(row.get("source_turn_index", -1)):
                errors.append(
                    f"{case_id}: prefix length does not match array turn index"
                )
            for turn in prefix:
                if set(turn) - {"speaker", "text", "turn_count"}:
                    errors.append(f"{case_id}: target metadata leaked into prefix")
                if turn.get("speaker") == "action" and not str(
                    turn.get("text", "")
                ).startswith("[ACTION] "):
                    errors.append(
                        f"{case_id}: prior action result text was not sanitized"
                    )
            assistant = (
                json.loads(row["messages"][-1].get("content", "{}"))
                if row.get("messages")
                else None
            )
            if assistant is not None and set(assistant) != {"action"}:
                errors.append(
                    f"{case_id}: training assistant target contains extra fields"
                )
        expected_main = [
            row for row in rows if row.get("conversation_policy_clean") is True
        ]
        if [canonical_json(row) for row in main_examples[split]] != [
            canonical_json(row) for row in expected_main
        ]:
            errors.append(
                f"{split}: main view is not exactly the policy-clean projection"
            )


def _validate_generated_counts(
    all_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    main_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    errors: list[str],
) -> None:
    for split, expected in EXPECTED_PRODUCT_DEFECT_INVENTORY.items():
        all_rows = all_examples[split]
        main_rows = main_examples[split]
        if (
            manifest.get("official_source")
            and len(all_rows) != expected["action_turn_count"]
        ):
            errors.append(
                f"{split}: actual all-action rows {len(all_rows)} != "
                f"{expected['action_turn_count']}"
            )
        reported = inventory["splits"][split]
        checks = {
            "example_count": len(all_rows),
            "policy_covered_example_count": sum(
                bool(row["policy_covered"]) for row in all_rows
            ),
            "policy_clean_conversation_count": len(
                {str(row["conversation_id"]) for row in main_rows}
            ),
            "subflow_action_examples": dict(
                sorted(Counter(str(row["subflow"]) for row in all_rows).items())
            ),
            "action_counts": dict(
                sorted(
                    Counter(
                        str(row["target_output"]["action"]) for row in all_rows
                    ).items()
                )
            ),
            "target_value_count": sum(
                len(row["raw_target_values"]) for row in all_rows
            ),
            "observable_target_value_count": sum(
                sum(bool(value) for value in row["value_observable"])
                for row in all_rows
            ),
        }
        for field, actual in checks.items():
            if reported.get(field) != actual:
                errors.append(
                    f"{split}: inventory {field} does not match generated rows"
                )
        if manifest.get("counts", {}).get("examples_all", {}).get(split) != len(
            all_rows
        ):
            errors.append(f"{split}: manifest all-action count mismatch")
        if manifest.get("counts", {}).get("examples_main", {}).get(split) != len(
            main_rows
        ):
            errors.append(f"{split}: manifest main-view count mismatch")


def _expected_artifact_paths() -> set[str]:
    return {
        *(
            f"examples/{split}_{view}.jsonl"
            for split in ("train", "dev", "test")
            for view in ("all_actions", "policy_clean")
        ),
        "arms/train_full_only.jsonl",
        "arms/train_no_policy_only.jsonl",
        "arms/train_random_mix.jsonl",
        "arms/train_staged_order.jsonl",
        *(f"arms/train_staged_{exposure}.jsonl" for exposure in EXPOSURES),
        "ms_swift/train_full_only.jsonl",
        "ms_swift/train_no_policy_only.jsonl",
        "ms_swift/train_random_mix.jsonl",
        "ms_swift/train_staged_order.jsonl",
        "eval/dev_policy_clean_eval.jsonl",
        "eval/dev_all_actions_eval.jsonl",
        "eval/test_policy_clean_eval.locked.jsonl",
        "eval/test_all_actions_eval.locked.jsonl",
        "inventory.json",
        "prompt_registry.json",
        "shortcut_baselines_dev.json",
    }


def _validate_artifact_manifest(
    root: Path, manifest: Mapping[str, Any], errors: list[str]
) -> None:
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != _expected_artifact_paths()
    ):
        errors.append(
            "build artifact manifest does not contain the exact frozen file set"
        )
        return
    for relative, expected in artifacts.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing generated artifact: {relative}")
            continue
        if path.stat().st_size != expected.get("size_bytes"):
            errors.append(f"artifact size mismatch: {relative}")
        if sha256_file(path) != expected.get("sha256"):
            errors.append(f"artifact SHA-256 mismatch: {relative}")
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
            if row_count != expected.get("row_count"):
                errors.append(f"artifact row-count mismatch: {relative}")


def _validate_split_isolation(
    examples: Mapping[str, Sequence[Mapping[str, Any]]], errors: list[str]
) -> None:
    for field in ("conversation_id", "conversation_hash"):
        values = {
            split: {str(row[field]) for row in rows} for split, rows in examples.items()
        }
        pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
        for left, right in pairs:
            overlap = values[left] & values[right]
            if overlap:
                errors.append(f"{field} overlap between {left}/{right}: {len(overlap)}")


def _validate_training_records(
    root: Path,
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    train_examples: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    order_seed: int,
    errors: list[str],
) -> None:
    if order_seed < 0:
        errors.append("training order seed is missing")
    examples = {str(row["case_id"]): row for row in train_examples}
    staged_by_id: dict[str, str] = {}
    random_by_id: dict[str, str] = {}
    for arm_name, rows in arms.items():
        for row in rows:
            case_id = str(row.get("case_id"))
            example = examples.get(case_id)
            if example is None:
                errors.append(f"{arm_name}: unknown training case {case_id}")
                continue
            exposure = str(row.get("prompt_exposure"))
            if exposure not in EXPOSURES:
                errors.append(f"{arm_name}/{case_id}: invalid exposure {exposure}")
                continue
            for key, value in example.items():
                if canonical_json(row.get(key)) != canonical_json(value):
                    errors.append(f"{arm_name}/{case_id}: source field changed: {key}")
                    break
            expected_messages = build_messages(
                example, exposure, registry, include_answer=True
            )
            if canonical_json(row.get("messages")) != canonical_json(expected_messages):
                errors.append(f"{arm_name}/{case_id}: rendered messages differ")
            expected_hashes = prompt_component_hashes(example, exposure, registry)
            if any(row.get(key) != value for key, value in expected_hashes.items()):
                errors.append(f"{arm_name}/{case_id}: prompt component hash differs")
            repetition = row.get("repetition")
            identity = f"{case_id}|{exposure}"
            if repetition is not None:
                identity += f"|repeat={repetition}"
            if row.get("record_id") != sha256_text(identity):
                errors.append(f"{arm_name}/{case_id}: record ID was not regenerated")
            try:
                assistant = json.loads(expected_messages[-1]["content"])
            except json.JSONDecodeError:
                errors.append(f"{arm_name}/{case_id}: assistant target is not JSON")
            else:
                if set(assistant) != {"action"}:
                    errors.append(
                        f"{arm_name}/{case_id}: assistant target is not action-only"
                    )
        _validate_ms_swift_projection(
            root / "ms_swift" / f"{arm_name}.jsonl",
            rows,
            arm_name,
            errors,
        )
        if arm_name == "train_staged_order":
            staged_by_id = {str(row["record_id"]): canonical_json(row) for row in rows}
        elif arm_name == "train_random_mix":
            random_by_id = {str(row["record_id"]): canonical_json(row) for row in rows}

    if staged_by_id != random_by_id:
        errors.append("Random-Mix and Staged records differ beyond ordering")
    for arm_name, exposure in (
        ("train_full_only", "full"),
        ("train_no_policy_only", "no_policy"),
    ):
        rows = arms[arm_name]
        expected_pairs = Counter(
            (str(example["case_id"]), repetition)
            for example in train_examples
            for repetition in range(len(EXPOSURES))
        )
        actual_pairs = Counter(
            (str(row["case_id"]), row.get("repetition")) for row in rows
        )
        if actual_pairs != expected_pairs:
            errors.append(f"{arm_name}: repetition coverage differs")
        if any(row.get("prompt_exposure") != exposure for row in rows):
            errors.append(f"{arm_name}: contains another exposure")

    staged = arms["train_staged_order"]
    for exposure in EXPOSURES:
        expected = [row for row in staged if row["prompt_exposure"] == exposure]
        actual = _load_jsonl(root / "arms" / f"train_staged_{exposure}.jsonl")
        if [canonical_json(row) for row in actual] != [
            canonical_json(row) for row in expected
        ]:
            errors.append(f"staged {exposure} projection differs")


def _validate_ms_swift_projection(
    path: Path,
    arm_rows: Sequence[Mapping[str, Any]],
    arm_name: str,
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"{arm_name}: ms-swift projection is missing")
        return
    with path.open("r", encoding="utf-8") as handle:
        swift_lines = (json.loads(line) for line in handle if line.strip())
        for index, (swift_row, arm_row) in enumerate(
            zip_longest(swift_lines, arm_rows), start=1
        ):
            if swift_row is None or arm_row is None:
                errors.append(f"{arm_name}: ms-swift row count differs")
                break
            if set(swift_row) != {"messages"}:
                errors.append(f"{arm_name}: ms-swift row {index} has extra fields")
                break
            if canonical_json(swift_row["messages"]) != canonical_json(
                arm_row["messages"]
            ):
                errors.append(f"{arm_name}: ms-swift row {index} differs")
                break


def _validate_eval_panel(
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    errors: list[str],
    split: str,
) -> None:
    example_by_id = {str(row["case_id"]): row for row in examples}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pairing_keys = set()
    for row in rows:
        groups[str(row["case_id"])].append(row)
        key = str(row["pairing_key"])
        if key in pairing_keys:
            errors.append(f"{split}: duplicate pairing key {key}")
        pairing_keys.add(key)
    if set(groups) != {str(row["case_id"]) for row in examples}:
        errors.append(f"{split}: eval case set differs from source examples")
    for case_id, variants in groups.items():
        if Counter(row["prompt_exposure"] for row in variants) != Counter(EXPOSURES):
            errors.append(f"{split}/{case_id}: exposure pairing is incomplete")
            continue
        for field in (
            "dialogue_prefix_hash",
            "target_hash",
            "user_prompt_sha256",
            "common_contract_sha256",
            "flow",
            "subflow",
        ):
            if len({canonical_json(row[field]) for row in variants}) != 1:
                errors.append(f"{split}/{case_id}: paired field changed: {field}")
        example = example_by_id.get(case_id)
        if example is None:
            continue
        for row in variants:
            exposure = str(row["prompt_exposure"])
            if row.get("pairing_key") != f"{case_id}|{exposure}":
                errors.append(f"{split}/{case_id}: pairing key was not regenerated")
            for key, value in example.items():
                if canonical_json(row.get(key)) != canonical_json(value):
                    errors.append(
                        f"{split}/{case_id}: eval source field changed: {key}"
                    )
                    break
            expected_messages = build_messages(
                example, exposure, registry, include_answer=False
            )
            if canonical_json(row.get("messages")) != canonical_json(expected_messages):
                errors.append(f"{split}/{case_id}: eval messages differ")
            expected_hashes = prompt_component_hashes(example, exposure, registry)
            if any(row.get(key) != value for key, value in expected_hashes.items()):
                errors.append(f"{split}/{case_id}: eval prompt hash differs")


def _validate_official_inventory(
    inventory: Mapping[str, Any], errors: list[str]
) -> None:
    for split, expected in EXPECTED_PRODUCT_DEFECT_INVENTORY.items():
        actual = inventory["splits"][split]
        checks = {
            "conversation_count": actual.get("conversation_count"),
            "action_turn_count": actual.get("example_count"),
            "policy_covered_action_count": actual.get("policy_covered_example_count"),
            "policy_clean_conversation_count": actual.get(
                "policy_clean_conversation_count"
            ),
        }
        for field, expected_value in expected.items():
            if checks[field] != expected_value:
                errors.append(
                    f"official {split}/{field}: {checks[field]} != {expected_value}"
                )


def _low_conversation_cluster_cells(
    examples: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in ("dev", "test"):
        counts: dict[str, set[str]] = defaultdict(set)
        for row in examples[split]:
            counts[str(row["subflow"])].add(str(row["conversation_id"]))
        low = {key: len(value) for key, value in counts.items() if len(value) < 20}
        if low:
            output[split] = dict(sorted(low.items()))
    return output


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
