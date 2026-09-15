"""Integrity-check and merge ms-swift responses into ABCD eval records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_annealing.abcd_posttraining.dataset import canonical_json
from skill_annealing.abcd_posttraining.evaluator import authorize_prediction_path
from skill_annealing.abcd_posttraining.source import sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_predictions(
    eval_path: Path,
    swift_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    eval_rows = load_jsonl(eval_path)
    swift_rows = load_jsonl(swift_path)
    if len(eval_rows) != len(swift_rows):
        raise ValueError(
            f"record count mismatch: eval={len(eval_rows)}, swift={len(swift_rows)}"
        )
    pairing_keys = [str(row["pairing_key"]) for row in eval_rows]
    if len(set(pairing_keys)) != len(pairing_keys):
        raise ValueError("eval file contains duplicate pairing keys")

    merged = []
    for index, (eval_row, swift_row) in enumerate(
        zip(eval_rows, swift_rows, strict=True), start=1
    ):
        response = swift_row.get("response")
        if not isinstance(response, str):
            raise ValueError(f"swift row {index} has no string response")
        messages = swift_row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"swift row {index} has no messages for order audit")
        prompt_messages: Sequence[Mapping[str, Any]] = messages
        if messages and messages[-1].get("role") == "assistant":
            assistant_content = messages[-1].get("content")
            if assistant_content != response:
                raise ValueError(
                    f"swift row {index} assistant content differs from response"
                )
            prompt_messages = messages[:-1]
        normalized_prompt = [
            {"role": message.get("role"), "content": message.get("content")}
            for message in prompt_messages
        ]
        if canonical_json(normalized_prompt) != canonical_json(eval_row["messages"]):
            raise ValueError(f"swift row {index} prompt/order mismatch")
        merged.append({**eval_row, "prediction": response})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(canonical_json(row) + "\n")
    ordered_pairing_hash = hashlib.sha256(
        canonical_json(pairing_keys).encode("utf-8")
    ).hexdigest()
    return {
        "status": "pass",
        "count": len(merged),
        "output": str(output_path),
        "ordered_pairing_key_sha256": ordered_pairing_hash,
        "eval_file_sha256": sha256_file(eval_path),
        "swift_result_file_sha256": sha256_file(swift_path),
        "output_file_sha256": sha256_file(output_path),
        "positional_order_verified_from_messages": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--swift-result-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--final-eval-manifest")
    args = parser.parse_args()
    authorize_prediction_path(
        args.eval_file,
        allow_locked_test=args.allow_locked_test,
        final_eval_manifest=args.final_eval_manifest,
    )
    report = merge_predictions(
        Path(args.eval_file),
        Path(args.swift_result_file),
        Path(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
