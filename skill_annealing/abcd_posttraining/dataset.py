"""Load ABCD and extract leakage-safe action-turn examples."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import (
    BUTTON_ACTION_OVERRIDES,
    FLOW_ID,
    FLOW_TITLE,
    PRODUCT_DEFECT_SUBFLOWS,
    SUBFLOW_TITLES,
    TASK_ID,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_abcd(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else source.open
    with opener(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not {"train", "dev", "test"} <= set(payload):
        raise ValueError("ABCD source must contain train/dev/test conversation splits")
    return payload


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def extract_action_examples(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    flow: str = FLOW_ID,
    subflows: Sequence[str] = PRODUCT_DEFECT_SUBFLOWS,
    policy_actions: Mapping[str, set[str]] | None = None,
    action_vocabulary: Sequence[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected_subflows = set(subflows)
    known_actions = set(action_vocabulary or ())
    examples: dict[str, list[dict[str, Any]]] = {}
    inventory: dict[str, Any] = {
        "flow": flow,
        "subflows": list(subflows),
        "splits": {},
        "mapping_eligible": 0,
        "mapping_emitted": 0,
        "filtered_reasons": {},
    }
    filtered = Counter()

    for split in ("train", "dev", "test"):
        rows: list[dict[str, Any]] = []
        conversations = 0
        eligible = 0
        subflow_conversations = Counter()
        subflow_actions = Counter()
        action_counts = Counter()
        value_count = 0
        observable_value_count = 0
        policy_covered_count = 0
        clean_conversation_count = 0

        for conversation in dataset[split]:
            scenario = conversation.get("scenario") or {}
            if (
                scenario.get("flow") != flow
                or scenario.get("subflow") not in selected_subflows
            ):
                continue
            conversations += 1
            subflow = str(scenario["subflow"])
            subflow_conversations[subflow] += 1
            turns = conversation.get("delexed")
            if not isinstance(turns, list):
                filtered["missing_delexed_conversation"] += 1
                continue
            conversation_id = str(conversation.get("convo_id"))
            conversation_actions = _conversation_action_labels(turns)
            subflow_policy_actions = (
                set(policy_actions.get(subflow, set())) if policy_actions else set()
            )
            conversation_policy_clean = bool(
                subflow_policy_actions
                and conversation_actions
                and all(
                    action in subflow_policy_actions for action in conversation_actions
                )
            )
            if conversation_policy_clean:
                clean_conversation_count += 1
            sanitized_transcript = _sanitized_transcript(turns)
            transcript_hash = sha256_text(canonical_json(sanitized_transcript))
            prior_actions: list[str] = []
            public_prefix: list[dict[str, Any]] = []

            for turn_index, turn in enumerate(turns):
                targets = turn.get("targets") if isinstance(turn, Mapping) else None
                speaker = (
                    str(turn.get("speaker", "unknown"))
                    if isinstance(turn, Mapping)
                    else "unknown"
                )
                if not isinstance(targets, list) or len(targets) < 5:
                    if speaker != "action":
                        public_prefix.append(_public_dialogue_turn(turn))
                    continue
                _, next_step, action, values, _ = targets[:5]
                is_action_target = speaker == "action" and next_step == "take_action"
                if not is_action_target:
                    if speaker != "action":
                        public_prefix.append(_public_dialogue_turn(turn))
                    continue
                eligible += 1
                inventory["mapping_eligible"] += 1
                if not isinstance(action, str) or not action.strip():
                    filtered["missing_action_label"] += 1
                    continue
                if known_actions and action not in known_actions:
                    filtered["action_outside_ontology"] += 1
                    continue
                if values is None:
                    values = []
                if not isinstance(values, list) or not all(
                    isinstance(value, (str, int, float, bool)) for value in values
                ):
                    filtered["invalid_values"] += 1
                    continue

                raw_target_values = [str(value) for value in values]
                normalized_target_values = normalize_values(values)
                prefix = list(public_prefix)
                prefix_text = "\n".join(str(item["text"]) for item in prefix)
                flags = [
                    value_is_observable(value, prefix_text)
                    for value in raw_target_values
                ]
                value_count += len(flags)
                observable_value_count += sum(flags)
                target_output = {"action": action}
                policy_covered = bool(action in subflow_policy_actions)
                if policy_covered:
                    policy_covered_count += 1
                turn_count = turn.get("turn_count", turn_index + 1)
                case_id = f"abcd_{split}_{conversation_id}_i{turn_index}"
                record = {
                    "case_id": case_id,
                    "task_id": TASK_ID,
                    "split": split,
                    "conversation_id": conversation_id,
                    "conversation_hash": transcript_hash,
                    "source_turn_index": turn_index,
                    "source_turn_count": turn_count,
                    "flow": flow,
                    "subflow": subflow,
                    "dialogue_prefix": prefix,
                    "dialogue_prefix_hash": sha256_text(canonical_json(prefix)),
                    "prior_actions": list(prior_actions),
                    "prior_action_count": len(prior_actions),
                    "last_action": prior_actions[-1] if prior_actions else None,
                    "target_output": target_output,
                    "target_hash": sha256_text(canonical_json(target_output)),
                    "raw_target_values": raw_target_values,
                    "normalized_target_values": normalized_target_values,
                    "target_turn_text_hash": sha256_text(str(turn.get("text", ""))),
                    "value_observable": flags,
                    "policy_covered": policy_covered,
                    "conversation_policy_clean": conversation_policy_clean,
                }
                rows.append(record)
                prior_actions.append(action)
                public_prefix.append(
                    {
                        "speaker": "action",
                        "text": f"[ACTION] {action}",
                        "turn_count": turn_count,
                    }
                )
                subflow_actions[subflow] += 1
                action_counts[action] += 1
                inventory["mapping_emitted"] += 1

        rows.sort(key=lambda row: (row["conversation_id"], row["source_turn_index"]))
        examples[split] = rows
        inventory["splits"][split] = {
            "conversation_count": conversations,
            "action_turn_eligible": eligible,
            "example_count": len(rows),
            "policy_covered_example_count": policy_covered_count,
            "policy_covered_rate": policy_covered_count / len(rows) if rows else 0.0,
            "policy_clean_conversation_count": clean_conversation_count,
            "subflow_conversations": dict(sorted(subflow_conversations.items())),
            "subflow_action_examples": dict(sorted(subflow_actions.items())),
            "action_counts": dict(sorted(action_counts.items())),
            "target_value_count": value_count,
            "observable_target_value_count": observable_value_count,
            "target_value_observable_rate": (
                observable_value_count / value_count if value_count else 1.0
            ),
        }

    inventory["mapping_coverage"] = (
        inventory["mapping_emitted"] / inventory["mapping_eligible"]
        if inventory["mapping_eligible"]
        else 0.0
    )
    inventory["filtered_reasons"] = dict(sorted(filtered.items()))
    total_values = sum(
        split["target_value_count"] for split in inventory["splits"].values()
    )
    total_observable = sum(
        split["observable_target_value_count"] for split in inventory["splits"].values()
    )
    inventory["target_value_observable_rate"] = (
        total_observable / total_values if total_values else 1.0
    )
    return examples, inventory


def flatten_action_vocabulary(ontology: Mapping[str, Any]) -> tuple[str, ...]:
    actions = ontology.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError("ontology.actions is missing")
    vocabulary = {
        str(action)
        for category in actions.values()
        if isinstance(category, Mapping)
        for action in category
    }
    if not vocabulary:
        raise ValueError("ontology action vocabulary is empty")
    return tuple(sorted(vocabulary))


def build_policy_action_sets(
    guidelines: Mapping[str, Any], action_vocabulary: Sequence[str]
) -> dict[str, set[str]]:
    allowed = set(action_vocabulary)
    output: dict[str, set[str]] = {}
    flow = guidelines[FLOW_TITLE]
    for subflow, title in SUBFLOW_TITLES.items():
        actions: set[str] = set()
        guideline = flow["subflows"][title]
        for step in guideline.get("actions", []):
            texts = [str(step.get("button", "")), str(step.get("text", ""))]
            texts.extend(map(str, step.get("subtext", [])))
            candidates = [texts[0]]
            for text in texts[1:]:
                candidates.extend(re.findall(r"\[([^\]]+)\]", text))
            for candidate in candidates:
                action = button_to_action(candidate)
                if action in allowed:
                    actions.add(action)
        output[subflow] = actions
    return output


def button_to_action(button: str) -> str:
    normalized = re.sub(r"\s+", " ", button.casefold()).strip()
    if normalized in BUTTON_ACTION_OVERRIDES:
        return BUTTON_ACTION_OVERRIDES[normalized]
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def normalize_values(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def value_is_observable(value: str, prefix_text: str) -> bool:
    if not value:
        return True
    haystack = _match_normalize(prefix_text)
    candidates = {
        _match_normalize(value),
        _match_normalize(value.replace("_", " ")),
    }
    return any(candidate and candidate in haystack for candidate in candidates)


def _match_normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _public_dialogue_turn(turn: Any) -> dict[str, Any]:
    if not isinstance(turn, Mapping):
        raise ValueError("delexed turns must be JSON objects")
    return {
        "speaker": str(turn.get("speaker", "unknown")),
        "text": str(turn.get("text", "")),
        "turn_count": turn.get("turn_count"),
    }


def _conversation_action_labels(turns: Sequence[Any]) -> list[str]:
    labels = []
    for turn in turns:
        if not isinstance(turn, Mapping) or turn.get("speaker") != "action":
            continue
        targets = turn.get("targets")
        if (
            isinstance(targets, list)
            and len(targets) >= 3
            and targets[1] == "take_action"
            and isinstance(targets[2], str)
            and targets[2]
        ):
            labels.append(targets[2])
    return labels


def _sanitized_transcript(turns: Sequence[Any]) -> list[dict[str, Any]]:
    transcript = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("delexed turns must be JSON objects")
        if turn.get("speaker") == "action":
            targets = turn.get("targets")
            action = (
                targets[2]
                if isinstance(targets, list)
                and len(targets) >= 3
                and isinstance(targets[2], str)
                else "unknown"
            )
            transcript.append(
                {
                    "speaker": "action",
                    "text": f"[ACTION] {action}",
                    "turn_count": turn.get("turn_count"),
                }
            )
        else:
            transcript.append(_public_dialogue_turn(turn))
    return transcript
