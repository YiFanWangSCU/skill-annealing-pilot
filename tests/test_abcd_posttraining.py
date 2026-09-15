from __future__ import annotations

import json
import sys
from collections import Counter

import pytest

from scripts.merge_abcd_swift_predictions import merge_predictions

from skill_annealing.abcd_posttraining.baselines import (
    evaluate_shortcut_baselines,
)
from skill_annealing.abcd_posttraining.builder import (
    build_bundle,
    build_training_arms,
)
from skill_annealing.abcd_posttraining.constants import (
    EXPECTED_SOURCE_METADATA,
    EXPOSURES,
    PRODUCT_DEFECT_SUBFLOWS,
    SOURCE_COMMIT,
    SOURCE_FILES,
    SOURCE_REPOSITORY,
    SUBFLOW_TITLES,
)
from skill_annealing.abcd_posttraining.dataset import (
    build_policy_action_sets,
    extract_action_examples,
    flatten_action_vocabulary,
)
from skill_annealing.abcd_posttraining import evaluator as abcd_evaluator
from skill_annealing.abcd_posttraining.evaluator import (
    authorize_prediction_path,
    cluster_bootstrap_by_exposure,
    cluster_bootstrap_interval,
    evaluate_by_exposure,
    evaluate_records,
    paired_cluster_bootstrap_difference_by_exposure,
    paired_cluster_bootstrap_gap_of_gaps,
    score_record,
)
from skill_annealing.abcd_posttraining.prompts import (
    build_prompt_registry,
    build_system_prompt,
    build_user_prompt,
)
from skill_annealing.abcd_posttraining.validator import (
    reject_locked_test_path,
    validate_bundle,
)
from skill_annealing.abcd_posttraining.source import verify_sources


def _ontology() -> dict:
    return {
        "actions": {
            "kb_query": {"validate-purchase": []},
            "interaction": {
                "pull-up-account": [],
                "record-reason": [],
                "enter-details": [],
                "offer-refund": [],
                "membership": [],
                "notify-team": [],
                "update-order": [],
            },
        }
    }


def _guidelines() -> dict:
    final_buttons = {
        "refund_initiate": "Offer Refund",
        "refund_update": "Record Reason",
        "refund_status": "Notify Internal Team",
        "return_stain": "Membership Privileges",
        "return_color": "Membership Privileges",
        "return_size": "Membership Privileges",
    }
    subflows = {}
    for subflow, title in SUBFLOW_TITLES.items():
        final_button = final_buttons[subflow]
        subtext = ["If needed, then continue."]
        if subflow == "refund_status":
            subtext.append("If the method changes, use [Update Order].")
        subflows[title] = {
            "instructions": [f"Resolve {title} safely.", "Close politely."],
            "actions": [
                {
                    "type": "interaction",
                    "button": "Pull up Account",
                    "text": "Load the account.",
                    "subtext": [],
                },
                {
                    "type": "kb query",
                    "button": "Validate Purchase",
                    "text": "Validate the order.",
                    "subtext": [],
                },
                {
                    "type": "interaction",
                    "button": final_button,
                    "text": "Perform the workflow-specific action.",
                    "subtext": subtext,
                },
            ],
        }
    return {
        "Product Defect": {
            "description": "refunds and returns",
            "subflows": subflows,
        }
    }


def _target(subflow: str, action: str, values: list[str]) -> list:
    return [subflow, "take_action", action, values, None]


def _conversation(conversation_id: int, subflow: str) -> dict:
    final_actions = {
        "refund_initiate": "offer-refund",
        "refund_update": "record-reason",
        "refund_status": "notify-team",
        "return_stain": "membership",
        "return_color": "membership",
        "return_size": "membership",
    }
    turns = [
        {
            "speaker": "customer",
            "text": f"Please help with order case {conversation_id}.",
            "turn_count": 2,
            "targets": [subflow, "retrieve_utterance", None, [], 0],
            "candidates": [0],
        },
        {
            "speaker": "action",
            "text": "Account data for a hidden real customer.",
            "turn_count": 4,
            "targets": _target(subflow, "pull-up-account", ["real-account-123"]),
            "candidates": [],
        },
        {
            "speaker": "customer",
            "text": "My order id is <order_id>.",
            "turn_count": 5,
            "targets": [subflow, "retrieve_utterance", None, [], 0],
            "candidates": [0],
        },
        {
            "speaker": "action",
            "text": "The real order was validated.",
            "turn_count": 7,
            "targets": _target(subflow, "validate-purchase", ["987654"]),
            "candidates": [],
        },
        {
            "speaker": "agent",
            "text": "I can continue with the request.",
            "turn_count": 8,
            "targets": [subflow, "retrieve_utterance", None, [], 0],
            "candidates": [0],
        },
        {
            "speaker": "action",
            "text": "The final action completed.",
            "turn_count": 10,
            "targets": _target(subflow, final_actions[subflow], []),
            "candidates": [],
        },
    ]
    return {
        "convo_id": conversation_id,
        "scenario": {
            "flow": "product_defect",
            "subflow": subflow,
            "personal": {"secret_future_field": "must-not-leak"},
        },
        "original": [],
        "delexed": turns,
    }


def _dataset() -> dict:
    output = {"train": [], "dev": [], "test": []}
    identifier = 1
    for split, repetitions in (("train", 2), ("dev", 1), ("test", 1)):
        for subflow in PRODUCT_DEFECT_SUBFLOWS:
            for _ in range(repetitions):
                output[split].append(_conversation(identifier, subflow))
                identifier += 1
    return output


def _extracted():
    ontology = _ontology()
    guidelines = _guidelines()
    vocabulary = flatten_action_vocabulary(ontology)
    policy_actions = build_policy_action_sets(guidelines, vocabulary)
    examples, inventory = extract_action_examples(
        _dataset(),
        policy_actions=policy_actions,
        action_vocabulary=vocabulary,
    )
    return examples, inventory, guidelines, vocabulary


def test_action_extraction_uses_array_index_and_sanitizes_prior_actions() -> None:
    examples, inventory, _, _ = _extracted()
    first, second = examples["train"][:2]

    assert first["source_turn_index"] == 1
    assert first["source_turn_count"] == 4
    assert first["case_id"].endswith("_i1")
    assert len(first["dialogue_prefix"]) == 1
    assert second["source_turn_index"] == 3
    assert second["dialogue_prefix"][1]["text"] == "[ACTION] pull-up-account"
    assert "hidden real customer" not in json.dumps(second["dialogue_prefix"])
    assert set(first["target_output"]) == {"action"}
    assert first["raw_target_values"] == ["real-account-123"]
    assert first["value_observable"] == [False]
    assert "scenario" not in first
    assert inventory["mapping_coverage"] == 1.0


def test_raw_values_preserve_order_duplicates_and_empty_strings() -> None:
    dataset = _dataset()
    dataset["train"][0]["delexed"][1]["targets"][3] = ["x", "", "x", " y "]
    vocabulary = flatten_action_vocabulary(_ontology())
    examples, _ = extract_action_examples(
        dataset,
        policy_actions=build_policy_action_sets(_guidelines(), vocabulary),
        action_vocabulary=vocabulary,
    )
    first = examples["train"][0]
    assert first["raw_target_values"] == ["x", "", "x", " y "]
    assert first["normalized_target_values"] == ["x", "y"]


def test_policy_registry_handles_abcd_button_aliases_and_hidden_action() -> None:
    _, _, guidelines, vocabulary = _extracted()
    registry = build_prompt_registry(guidelines, vocabulary)

    assert "membership" in registry["policy_actions"]["return_size"]
    assert "notify-team" in registry["policy_actions"]["refund_status"]
    assert "update-order" in registry["policy_actions"]["refund_status"]
    assert "membership-privileges" not in registry["global_action_vocabulary"]


def test_all_exposures_keep_schema_role_and_oracle_workflow() -> None:
    examples, _, guidelines, vocabulary = _extracted()
    registry = build_prompt_registry(guidelines, vocabulary)
    example = examples["dev"][0]
    prompts = {
        exposure: build_system_prompt(registry, example["subflow"], exposure)
        for exposure in EXPOSURES
    }

    assert len(prompts) == 4
    assert all(
        '{"action":"<official-action-code>"}' in text for text in prompts.values()
    )
    assert all("values" not in text for text in prompts.values())
    assert "Complete workflow procedure" in prompts["full"]
    assert "Ordered executable actions" in prompts["partial"]
    assert "Executable action set (unordered)" in prompts["minimal"]
    assert "Workflow objective" not in prompts["no_policy"]
    user_prompt = build_user_prompt(example)
    assert "flow=product_defect" in user_prompt
    assert f"subflow={example['subflow']}" in user_prompt
    assert "routing is out of scope" in user_prompt


def test_training_arms_match_records_and_optimizer_counts() -> None:
    examples, _, guidelines, vocabulary = _extracted()
    registry = build_prompt_registry(guidelines, vocabulary)
    arms = build_training_arms(examples["train"], registry, order_seed=17)

    staged_ids = [row["record_id"] for row in arms["train_staged_order"]]
    random_ids = [row["record_id"] for row in arms["train_random_mix"]]
    assert Counter(staged_ids) == Counter(random_ids)
    assert staged_ids != random_ids
    assert len(arms["train_full_only"]) == len(staged_ids)
    assert len(arms["train_no_policy_only"]) == len(staged_ids)
    assert Counter(
        row["prompt_exposure"] for row in arms["train_staged_order"]
    ) == Counter({exposure: len(examples["train"]) for exposure in EXPOSURES})
    assert all(
        set(json.loads(row["messages"][-1]["content"])) == {"action"}
        for row in arms["train_staged_order"]
    )


def test_evaluator_separates_raw_action_from_strict_contract() -> None:
    examples, _, _, vocabulary = _extracted()
    example = examples["dev"][0]
    extra = score_record(
        {
            **example,
            "predicted_output": {"action": example["target_output"]["action"], "x": 1},
        },
        vocabulary,
    )
    duplicate = score_record(
        {
            **example,
            "prediction": '{"action":"pull-up-account","action":"pull-up-account"}',
        },
        vocabulary,
    )

    assert extra["action_correct"] == 1.0
    assert extra["schema_valid"] == 0.0
    assert duplicate["json_valid"] == 0.0


def test_shortcut_gate_detects_fixed_workflow_position() -> None:
    examples, _, _, vocabulary = _extracted()
    report = evaluate_shortcut_baselines(examples["train"], examples["dev"], vocabulary)

    assert report["subflow_step_action_accuracy"] == 1.0
    assert report["overall_action_metric_decisive"] is False
    assert report["primary_tail_metric_decisive"] is False


def test_fixture_bundle_passes_static_gates_but_blocks_strong_claim(tmp_path) -> None:
    output = tmp_path / "build"
    build_bundle(
        tmp_path / "unused-raw",
        output,
        dataset_payload=_dataset(),
        guidelines_payload=_guidelines(),
        ontology_payload=_ontology(),
    )
    report = validate_bundle(output)

    assert report["local_static_valid"], report["errors"]
    assert report["gpu_smoke_ready"] is True
    assert report["formal_claim_ready"] is False
    assert report["primary_tail_metric_decisive"] is False
    assert report["oracle_dev_policy_clean"]["strict_action_accuracy"] == 1.0


def test_validator_detects_tampered_ms_swift_and_dropped_rows(tmp_path) -> None:
    output = tmp_path / "build"
    build_bundle(
        tmp_path / "unused-raw",
        output,
        dataset_payload=_dataset(),
        guidelines_payload=_guidelines(),
        ontology_payload=_ontology(),
    )
    swift_path = output / "ms_swift" / "train_random_mix.jsonl"
    swift_lines = swift_path.read_text(encoding="utf-8").splitlines()
    swift_lines[0] = json.dumps({"messages": []})
    swift_path.write_text("\n".join(swift_lines) + "\n", encoding="utf-8")
    report = validate_bundle(output)
    assert report["local_static_valid"] is False
    assert any("train_random_mix" in error for error in report["errors"])

    output_2 = tmp_path / "build_dropped"
    build_bundle(
        tmp_path / "unused-raw-2",
        output_2,
        dataset_payload=_dataset(),
        guidelines_payload=_guidelines(),
        ontology_payload=_ontology(),
    )
    main_path = output_2 / "examples" / "dev_policy_clean.jsonl"
    main_lines = main_path.read_text(encoding="utf-8").splitlines()
    main_path.write_text("\n".join(main_lines[:-1]) + "\n", encoding="utf-8")
    report = validate_bundle(output_2)
    assert report["local_static_valid"] is False
    assert any(
        "main view" in error or "artifact" in error for error in report["errors"]
    )


def test_source_manifest_requires_exact_frozen_file_set(tmp_path) -> None:
    assert set(EXPECTED_SOURCE_METADATA) == set(SOURCE_FILES)
    manifest = {
        "commit": SOURCE_COMMIT,
        "repository": f"https://github.com/{SOURCE_REPOSITORY}",
        "files": {},
    }
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="file set"):
        verify_sources(tmp_path)


def test_cluster_bootstrap_uses_conversations() -> None:
    examples, _, _, vocabulary = _extracted()
    rows = [{**row, "oracle": True} for row in examples["dev"]]
    report = cluster_bootstrap_interval(rows, vocabulary, samples=20, seed=9)
    assert report["cluster_key"] == "conversation_id"
    assert report["estimate"] == 1.0
    assert report["ci_95_low"] == 1.0


def _paired_prediction_panel(
    *, no_policy_correct: bool, minimal_correct: bool = False
) -> list[dict]:
    rows = []
    for conversation_id in ("conversation-a", "conversation-b"):
        case_id = f"case-{conversation_id}"
        for exposure in EXPOSURES:
            correct = exposure in {"full", "partial"}
            if exposure == "minimal":
                correct = minimal_correct
            elif exposure == "no_policy":
                correct = no_policy_correct
            rows.append(
                {
                    "task_id": "abcd_product_defect_ast",
                    "split": "dev",
                    "case_id": case_id,
                    "conversation_id": conversation_id,
                    "conversation_hash": f"conversation-hash-{conversation_id}",
                    "source_turn_index": 3,
                    "pairing_key": f"{case_id}|{exposure}",
                    "prompt_exposure": exposure,
                    "flow": "product_defect",
                    "subflow": "refund_initiate",
                    "dialogue_prefix_hash": f"dialogue-hash-{conversation_id}",
                    "policy_covered": True,
                    "target_output": {"action": "offer-refund"},
                    "target_hash": "target-hash-offer-refund",
                    "common_contract_sha256": "common-contract-hash",
                    "policy_sha256": f"policy-hash-{exposure}",
                    "user_prompt_sha256": f"user-prompt-hash-{conversation_id}",
                    "messages": [
                        {"role": "system", "content": f"policy {exposure}"},
                        {"role": "user", "content": f"case {conversation_id}"},
                    ],
                    "predicted_output": {
                        "action": "offer-refund" if correct else "record-reason"
                    },
                }
            )
    return rows


def test_by_exposure_bootstrap_does_not_pool_prompt_conditions() -> None:
    rows = _paired_prediction_panel(no_policy_correct=False)
    report = cluster_bootstrap_by_exposure(
        rows,
        ["offer-refund", "record-reason"],
        samples=20,
        seed=7,
    )

    assert report["resampling_unit"] == "conversation_id_within_exposure"
    assert report["by_exposure"]["full"]["estimate"] == 1.0
    assert report["by_exposure"]["partial"]["estimate"] == 1.0
    assert report["by_exposure"]["minimal"]["estimate"] == 0.0
    assert report["by_exposure"]["no_policy"]["estimate"] == 0.0
    assert report["retention"]["full_minus_no_policy"] == 1.0
    assert report["retention"]["ci_95_low"] == 1.0


def test_paired_comparison_reports_each_exposure_and_gap_of_gaps() -> None:
    candidate = _paired_prediction_panel(no_policy_correct=True)
    reference = _paired_prediction_panel(no_policy_correct=False)
    vocabulary = ["offer-refund", "record-reason"]
    per_exposure = paired_cluster_bootstrap_difference_by_exposure(
        candidate,
        reference,
        vocabulary,
        samples=20,
        seed=7,
    )
    gap_of_gaps = paired_cluster_bootstrap_gap_of_gaps(
        candidate,
        reference,
        vocabulary,
        samples=20,
        seed=7,
    )

    assert per_exposure["by_exposure"]["full"]["candidate_minus_reference"] == 0.0
    assert per_exposure["by_exposure"]["no_policy"]["candidate_minus_reference"] == 1.0
    assert gap_of_gaps["candidate_full_minus_no_policy"] == 0.0
    assert gap_of_gaps["reference_full_minus_no_policy"] == 1.0
    assert gap_of_gaps["candidate_minus_reference_gap"]["estimate"] == -1.0
    assert gap_of_gaps["no_policy_gain_minus_full_gain"]["estimate"] == 1.0
    assert gap_of_gaps["no_policy_gain_minus_full_gain"]["ci_95_low"] == 1.0


def test_retention_bootstrap_rejects_unmatched_exposure_cases() -> None:
    rows = _paired_prediction_panel(no_policy_correct=False)
    tampered = [
        row
        for row in rows
        if not (
            row["case_id"] == "case-conversation-b"
            and row["prompt_exposure"] == "no_policy"
        )
    ]
    with pytest.raises(ValueError, match="same case IDs"):
        cluster_bootstrap_by_exposure(
            tampered,
            ["offer-refund", "record-reason"],
            samples=20,
            seed=7,
        )


def test_by_exposure_bootstrap_requires_a_complete_panel() -> None:
    rows = [
        row
        for row in _paired_prediction_panel(no_policy_correct=True)
        if row["prompt_exposure"] != "minimal"
    ]
    with pytest.raises(ValueError, match="complete exposure panel"):
        cluster_bootstrap_by_exposure(
            rows,
            ["offer-refund", "record-reason"],
            samples=20,
            seed=7,
        )


def test_point_estimate_by_exposure_requires_a_complete_known_panel() -> None:
    rows = _paired_prediction_panel(no_policy_correct=True)
    missing = [row for row in rows if row["prompt_exposure"] != "minimal"]
    with pytest.raises(ValueError, match="complete exposure panel"):
        evaluate_by_exposure(missing, ["offer-refund", "record-reason"])

    unknown = [*rows, {**rows[0], "case_id": "extra", "prompt_exposure": "other"}]
    with pytest.raises(ValueError, match="unknown or missing prompt exposure"):
        evaluate_by_exposure(unknown, ["offer-refund", "record-reason"])


def test_paired_comparison_rejects_prompt_identity_mismatch() -> None:
    candidate = _paired_prediction_panel(no_policy_correct=True)
    reference = _paired_prediction_panel(no_policy_correct=False)
    reference[0] = {
        **reference[0],
        "messages": [
            {"role": "system", "content": "tampered policy"},
            {"role": "user", "content": "tampered user prompt"},
        ],
    }
    with pytest.raises(ValueError, match="messages"):
        paired_cluster_bootstrap_difference_by_exposure(
            candidate,
            reference,
            ["offer-refund", "record-reason"],
            samples=20,
            seed=7,
        )


def test_gap_of_gaps_rejects_same_exposure_identity_mismatch() -> None:
    candidate = _paired_prediction_panel(no_policy_correct=True)
    reference = _paired_prediction_panel(no_policy_correct=False)
    reference[0] = {**reference[0], "policy_sha256": "tampered-policy"}
    with pytest.raises(ValueError, match="policy_sha256"):
        paired_cluster_bootstrap_gap_of_gaps(
            candidate,
            reference,
            ["offer-refund", "record-reason"],
            samples=20,
            seed=7,
        )


def test_evaluator_cli_exposes_pre_registered_contrasts(
    tmp_path, monkeypatch, capsys
) -> None:
    candidate_path = tmp_path / "candidate.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    registry_path = tmp_path / "prompt_registry.json"
    output_path = tmp_path / "comparison.json"
    unlock_path = tmp_path / "final_eval_manifest.json"
    candidate = _paired_prediction_panel(no_policy_correct=True)
    reference = _paired_prediction_panel(no_policy_correct=False)
    for path, rows in ((candidate_path, candidate), (reference_path, reference)):
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    registry_path.write_text(
        json.dumps(
            {
                "global_action_vocabulary": ["offer-refund", "record-reason"],
                "policy_actions": {
                    "refund_initiate": ["offer-refund", "record-reason"]
                },
            }
        ),
        encoding="utf-8",
    )
    unlock_path.write_text(
        json.dumps(
            {
                "test_unlock": True,
                "model_selection_frozen": True,
                "run_id": "unit-test-authorized-path",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "abcd-evaluator",
            "--predictions",
            str(candidate_path),
            "--compare-predictions",
            str(reference_path),
            "--prompt-registry",
            str(registry_path),
            "--by-exposure",
            "--bootstrap-samples",
            "20",
            "--allow-locked-test",
            "--final-eval-manifest",
            str(unlock_path),
            "--output",
            str(output_path),
        ],
    )

    abcd_evaluator.main()
    capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        report["candidate"]["cluster_bootstrap"]["by_exposure"]["no_policy"]["estimate"]
        == 1.0
    )
    assert (
        report["paired_cluster_bootstrap"]["by_exposure"]["no_policy"][
            "candidate_minus_reference"
        ]
        == 1.0
    )
    assert (
        report["paired_cluster_bootstrap"]["retention_gap_of_gaps"][
            "no_policy_gain_minus_full_gain"
        ]["estimate"]
        == 1.0
    )


def test_locked_test_paths_are_rejected() -> None:
    with pytest.raises(PermissionError):
        reject_locked_test_path("eval/test_policy_clean_eval.locked.jsonl")
    reject_locked_test_path("eval/dev_policy_clean_eval.jsonl")
    with pytest.raises(PermissionError):
        authorize_prediction_path(
            "eval/test_policy_clean_eval.locked.jsonl",
            allow_locked_test=False,
            final_eval_manifest=None,
        )


def test_swift_merge_verifies_returned_prompt_order(tmp_path) -> None:
    eval_path = tmp_path / "dev_eval.jsonl"
    swift_path = tmp_path / "swift.jsonl"
    output_path = tmp_path / "merged.jsonl"
    eval_rows = [
        {
            "pairing_key": f"case-{index}|no_policy",
            "messages": [
                {"role": "system", "content": "contract"},
                {"role": "user", "content": f"dialogue {index}"},
            ],
        }
        for index in range(2)
    ]
    responses = ['{"action":"a"}', '{"action":"b"}']
    swift_rows = [
        {
            "response": response,
            "messages": [
                *[{**message, "loss": None} for message in eval_row["messages"]],
                {"role": "assistant", "content": response},
            ],
        }
        for eval_row, response in zip(eval_rows, responses, strict=True)
    ]
    for path, rows in ((eval_path, eval_rows), (swift_path, swift_rows)):
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    report = merge_predictions(eval_path, swift_path, output_path)
    assert report["status"] == "pass"
    assert report["positional_order_verified_from_messages"] is True

    swift_path.write_text(
        "\n".join(json.dumps(row) for row in reversed(swift_rows)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt/order"):
        merge_predictions(eval_path, swift_path, output_path)
