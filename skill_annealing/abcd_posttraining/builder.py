"""Build matched ABCD post-training arms and paired evaluation panels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baselines import evaluate_shortcut_baselines
from .constants import (
    DEFAULT_ORDER_SEED,
    EXPOSURES,
    PRODUCT_DEFECT_SUBFLOWS,
    TASK_ID,
    TRAINING_REPETITIONS,
)
from .dataset import (
    build_policy_action_sets,
    canonical_json,
    extract_action_examples,
    flatten_action_vocabulary,
    load_abcd,
    load_json,
    sha256_text,
)
from .prompts import build_messages, build_prompt_registry, prompt_component_hashes
from .source import sha256_file, verify_sources


def build_bundle(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    order_seed: int = DEFAULT_ORDER_SEED,
    dataset_payload: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    guidelines_payload: Mapping[str, Any] | None = None,
    ontology_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    official_source = dataset_payload is None
    if official_source:
        source_manifest = verify_sources(raw)
        dataset = load_abcd(raw / "abcd_v1.1.json.gz")
        guidelines = load_json(raw / "guidelines.json")
        ontology = load_json(raw / "ontology.json")
        source_manifest_hash = sha256_file(raw / "source_manifest.json")
    else:
        if guidelines_payload is None or ontology_payload is None:
            raise ValueError("fixture builds require dataset, guidelines, and ontology")
        source_manifest = {"commit": "synthetic-fixture"}
        dataset = dataset_payload
        guidelines = guidelines_payload
        ontology = ontology_payload
        source_manifest_hash = None
    action_vocabulary = flatten_action_vocabulary(ontology)
    policy_actions = build_policy_action_sets(guidelines, action_vocabulary)
    examples, inventory = extract_action_examples(
        dataset,
        policy_actions=policy_actions,
        action_vocabulary=action_vocabulary,
    )
    registry = build_prompt_registry(guidelines, action_vocabulary)

    example_dir = output / "examples"
    arm_dir = output / "arms"
    swift_dir = output / "ms_swift"
    eval_dir = output / "eval"
    for directory in (example_dir, arm_dir, swift_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    main_examples: dict[str, list[dict[str, Any]]] = {}
    for split, rows in examples.items():
        main_rows = [row for row in rows if row["conversation_policy_clean"]]
        main_examples[split] = main_rows
        _write_jsonl(example_dir / f"{split}_all_actions.jsonl", rows)
        _write_jsonl(example_dir / f"{split}_policy_clean.jsonl", main_rows)

    arms = build_training_arms(main_examples["train"], registry, order_seed=order_seed)
    for arm_name, rows in arms.items():
        _write_jsonl(arm_dir / f"{arm_name}.jsonl", rows)
        _write_jsonl(
            swift_dir / f"{arm_name}.jsonl",
            ({"messages": row["messages"]} for row in rows),
        )
    for exposure in EXPOSURES:
        stage_rows = [
            row
            for row in arms["train_staged_order"]
            if row["prompt_exposure"] == exposure
        ]
        _write_jsonl(arm_dir / f"train_staged_{exposure}.jsonl", stage_rows)

    dev_main = build_eval_panel(main_examples["dev"], registry)
    dev_all = build_eval_panel(examples["dev"], registry)
    test_main = build_eval_panel(main_examples["test"], registry)
    test_all = build_eval_panel(examples["test"], registry)
    _write_jsonl(eval_dir / "dev_policy_clean_eval.jsonl", dev_main)
    _write_jsonl(eval_dir / "dev_all_actions_eval.jsonl", dev_all)
    _write_jsonl(eval_dir / "test_policy_clean_eval.locked.jsonl", test_main)
    _write_jsonl(eval_dir / "test_all_actions_eval.locked.jsonl", test_all)

    shortcuts = evaluate_shortcut_baselines(
        main_examples["train"],
        main_examples["dev"],
        action_vocabulary,
    )
    inventory["policy_action_sets"] = {
        subflow: sorted(actions) for subflow, actions in policy_actions.items()
    }
    inventory["main_view"] = "conversation_policy_clean"
    inventory["main_split_counts"] = {
        split: len(rows) for split, rows in main_examples.items()
    }
    inventory["all_action_split_counts"] = {
        split: len(rows) for split, rows in examples.items()
    }
    _write_json(output / "inventory.json", inventory)
    _write_json(output / "prompt_registry.json", registry)
    _write_json(output / "shortcut_baselines_dev.json", shortcuts)

    artifact_paths = [
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
    ]
    artifacts = {
        relative: _artifact_metadata(output / relative) for relative in artifact_paths
    }

    manifest = {
        "task_id": TASK_ID,
        "no_routing": True,
        "oracle_workflow_fields": ["flow", "subflow"],
        "target_schema": {"action": "official_action_code"},
        "raw_values_used_as_target": False,
        "main_view": "conversation_policy_clean",
        "robustness_view": "all_actions",
        "subflows": list(PRODUCT_DEFECT_SUBFLOWS),
        "exposures": list(EXPOSURES),
        "order_seed": order_seed,
        "official_source": official_source,
        "source_manifest_sha256": source_manifest_hash,
        "source_commit": source_manifest["commit"],
        "counts": {
            "examples_main": {
                split: len(rows) for split, rows in main_examples.items()
            },
            "examples_all": {split: len(rows) for split, rows in examples.items()},
            "arms": {name: len(rows) for name, rows in arms.items()},
            "dev_policy_clean_eval": len(dev_main),
            "test_policy_clean_eval": len(test_main),
        },
        "arm_hashes": {
            name: {
                "ordered_sha256": _ordered_hash(rows),
                "record_multiset_sha256": _multiset_hash(rows),
            }
            for name, rows in arms.items()
        },
        "artifacts": artifacts,
        "test_lock": {
            "locked": True,
            "model_selection_permitted": False,
            "files": [
                "eval/test_policy_clean_eval.locked.jsonl",
                "eval/test_all_actions_eval.locked.jsonl",
            ],
        },
    }
    _write_json(output / "build_manifest.json", manifest)
    return {
        "output_dir": str(output),
        "source_commit": source_manifest["commit"],
        "inventory": inventory,
        "shortcuts": shortcuts,
        "manifest": manifest,
    }


def build_training_arms(
    train_examples: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    *,
    order_seed: int = DEFAULT_ORDER_SEED,
) -> dict[str, list[dict[str, Any]]]:
    mixed_records = [
        _render_training_record(example, exposure, registry)
        for exposure in EXPOSURES
        for example in train_examples
    ]
    staged = sorted(
        mixed_records,
        key=lambda row: (
            EXPOSURES.index(row["prompt_exposure"]),
            sha256_text(f"{order_seed}|stage|{row['record_id']}"),
        ),
    )
    random_mix = sorted(
        mixed_records,
        key=lambda row: sha256_text(f"{order_seed}|random|{row['record_id']}"),
    )
    full_only = [
        _render_training_record(example, "full", registry, repetition=repetition)
        for repetition in range(TRAINING_REPETITIONS)
        for example in train_examples
    ]
    no_policy_only = [
        _render_training_record(example, "no_policy", registry, repetition=repetition)
        for repetition in range(TRAINING_REPETITIONS)
        for example in train_examples
    ]
    return {
        "train_full_only": full_only,
        "train_no_policy_only": no_policy_only,
        "train_random_mix": random_mix,
        "train_staged_order": staged,
    }


def build_eval_panel(
    examples: Sequence[Mapping[str, Any]], registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        for exposure in EXPOSURES:
            hashes = prompt_component_hashes(example, exposure, registry)
            rows.append(
                {
                    **example,
                    "prompt_exposure": exposure,
                    "pairing_key": f"{example['case_id']}|{exposure}",
                    "messages": build_messages(
                        example, exposure, registry, include_answer=False
                    ),
                    **hashes,
                }
            )
    return rows


def _render_training_record(
    example: Mapping[str, Any],
    exposure: str,
    registry: Mapping[str, Any],
    *,
    repetition: int | None = None,
) -> dict[str, Any]:
    identity = f"{example['case_id']}|{exposure}"
    if repetition is not None:
        identity += f"|repeat={repetition}"
    return {
        **example,
        "prompt_exposure": exposure,
        "repetition": repetition,
        "record_id": sha256_text(identity),
        "messages": build_messages(example, exposure, registry, include_answer=True),
        **prompt_component_hashes(example, exposure, registry),
    }


def _ordered_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json([row["record_id"] for row in rows]))


def _multiset_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(sorted(row["record_id"] for row in rows)))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _artifact_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            metadata["row_count"] = sum(1 for line in handle if line.strip())
    return metadata
