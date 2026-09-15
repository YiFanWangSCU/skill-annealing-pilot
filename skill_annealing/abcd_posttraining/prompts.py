"""Render matched ABCD policy-exposure prompts without routing."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .constants import EXPOSURES, FLOW_TITLE, SUBFLOW_TITLES
from .dataset import (
    build_policy_action_sets,
    button_to_action,
    canonical_json,
    sha256_text,
)


COMMON_TASK_TEMPLATE = """You execute the already-selected ABCD customer-service workflow.
The workflow identifier below is oracle metadata. Do not infer, route, or change it.
Predict the next executable business action from the dialogue prefix.

Return exactly one JSON object with exactly this key:
{{"action":"<official-action-code>"}}
`action` must be one code from this global codebook:
{action_codebook}
Do not add prose, markdown, or extra keys."""


def build_prompt_registry(
    guidelines: Mapping[str, Any],
    action_vocabulary: Sequence[str],
) -> dict[str, Any]:
    common = COMMON_TASK_TEMPLATE.format(
        action_codebook=json.dumps(list(action_vocabulary), ensure_ascii=False)
    )
    policy_action_sets = build_policy_action_sets(guidelines, action_vocabulary)
    policies: dict[str, dict[str, str]] = {}
    for subflow, title in SUBFLOW_TITLES.items():
        guideline = guidelines[FLOW_TITLE]["subflows"][title]
        policies[subflow] = {
            exposure: render_policy(
                exposure,
                guideline,
                action_vocabulary,
                policy_action_sets[subflow],
            )
            for exposure in EXPOSURES
        }
    return {
        "common_task": common,
        "common_contract_sha256": sha256_text(common),
        "global_action_vocabulary": list(action_vocabulary),
        "policy_actions": {
            subflow: sorted(actions) for subflow, actions in policy_action_sets.items()
        },
        "policies": policies,
    }


def render_policy(
    exposure: str,
    guideline: Mapping[str, Any],
    action_vocabulary: Sequence[str],
    policy_actions: set[str],
) -> str:
    if exposure not in EXPOSURES:
        raise ValueError(f"unknown exposure: {exposure}")
    if exposure == "no_policy":
        return ""

    instructions = [str(item).strip() for item in guideline.get("instructions", [])]
    objective = instructions[0] if instructions else "Execute the selected workflow."
    steps = list(guideline.get("actions", []))
    allowed = set(action_vocabulary)
    executable = [
        button_to_action(str(step.get("button", "")))
        for step in steps
        if isinstance(step, Mapping)
    ]
    executable = [action for action in executable if action in allowed]

    if exposure == "minimal":
        return (
            "Workflow objective:\n"
            f"{objective}\n"
            "Executable action set (unordered):\n"
            f"{json.dumps(sorted(policy_actions), ensure_ascii=False)}"
        )

    if exposure == "partial":
        constraints = []
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            for text in [str(step.get("text", "")), *map(str, step.get("subtext", []))]:
                if _is_constraint(text):
                    constraints.append(_compact(text))
        lines = [
            "Workflow objective:",
            objective,
            "Ordered executable actions:",
            " -> ".join(executable),
        ]
        conditional_actions = sorted(policy_actions - set(executable))
        if conditional_actions:
            lines.extend(
                [
                    "Additional conditionally valid actions:",
                    json.dumps(conditional_actions, ensure_ascii=False),
                ]
            )
        if constraints:
            lines.extend(["Key conditions:", *[f"- {item}" for item in constraints]])
        return "\n".join(lines)

    lines = ["Complete workflow procedure:"]
    lines.extend(f"- {instruction}" for instruction in instructions)
    for index, step in enumerate(steps, start=1):
        button = str(step.get("button", ""))
        action = button_to_action(button)
        code = action if action in allowed else "non-action communication step"
        lines.append(f"{index}. [{code}] {_compact(str(step.get('text', '')))}")
        lines.extend(f"   - {_compact(str(item))}" for item in step.get("subtext", []))
    return "\n".join(lines)


def build_system_prompt(
    registry: Mapping[str, Any], subflow: str, exposure: str
) -> str:
    common = str(registry["common_task"])
    policy = str(registry["policies"][subflow][exposure])
    return common if not policy else f"{common}\n\n{policy}"


def build_user_prompt(example: Mapping[str, Any]) -> str:
    transcript = "\n".join(
        f"[{turn['speaker']}] {turn['text']}" for turn in example["dialogue_prefix"]
    )
    return (
        "Known workflow (oracle; routing is out of scope):\n"
        f"flow={example['flow']}\n"
        f"subflow={example['subflow']}\n\n"
        "Dialogue prefix:\n"
        f"{transcript}\n\n"
        "Predict the next executable business action."
    )


def build_messages(
    example: Mapping[str, Any],
    exposure: str,
    registry: Mapping[str, Any],
    *,
    include_answer: bool,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(registry, str(example["subflow"]), exposure),
        },
        {"role": "user", "content": build_user_prompt(example)},
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": canonical_json(example["target_output"])}
        )
    return messages


def prompt_component_hashes(
    example: Mapping[str, Any], exposure: str, registry: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "common_contract_sha256": str(registry["common_contract_sha256"]),
        "policy_sha256": sha256_text(
            str(registry["policies"][example["subflow"]][exposure])
        ),
        "user_prompt_sha256": sha256_text(build_user_prompt(example)),
    }


def _is_constraint(text: str) -> bool:
    lowered = text.casefold()
    markers = ("if ", "only ", "then ", "must ", "cannot", "valid ", "option", "within")
    return any(marker in lowered for marker in markers)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
