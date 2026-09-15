# Claude Code handoff

Read AGENTS.md, RUN_EXPERIMENT.md, docs/rule_exposure_protocol.md and
docs/pilot_registry.json. This branch contains the complete bounded pilot,
not merely a two-update starter. Use scripts.rule_exposure as the single entry.

## Execution workflow

1. Check Git version and working tree; preserve local edits. Read the current
   protocol/config and run default CPU tests and an oracle simulation.
2. Use a company-approved isolated environment. Install compatible PyTorch and
   matching torchvision, then configs/server_requirements.txt; run pip check.
   Do not modify shared packages, insert credentials into URLs or download a
   different model to make an error disappear.
3. Obtain only missing local model location and allocated physical GPU index
   from the operator. Do not assume GPU 0. Verify model identity and permissions.
   The package targets a complete local Qwen3.5-4B dense model; a differently
   named or typed model is not equivalent without a new explicit design.
4. When the user requests the documented campaign and supplies those fields,
   run the dry plan, then the full run --execute command. Continue through
   preflight, four matched training arms, five inference endpoints and report
   automatically. Do not ask for repeated approval between these stages.
5. Report summary/report.md, metrics.csv and summary.json. Include Git version,
   actual model identity, runtime versions and failures in a sanitized handoff.
   Never turn the CPU simulation scores into model results.

The user's target is one company A800, 80 GB. Physical availability and software
compatibility still need server-side checks; no A800 success is pre-certified.

## Failure and research boundaries

Use status for progress. Re-running the same command skips verified completed
stages. Preserve failed attempts and logs; explicit --retry-failed is only for
a resolved environment fault, at most two attempts per stage within the total
budget. It is not a quality retry. A code/config/model change requires a fresh
campaign and a recorded deviation; do not mix endpoints across versions.

Default budget is in configs/rule_exposure_pilot.json, not in ad hoc commands.
Do not launch the old ABCD 15-run proposal, add GRPO, enlarge datasets or search
hyperparameters automatically. Do not alter frozen labels, metrics or evaluation
policies after reading scores. One seed and synthetic facts cannot establish a
publishable or deployable method; an exact rule engine is a relevant comparator.

If essential server information is missing, ask only for that missing field.
If a failure needs new authority or a design change, report the concrete blocker.
Normal in-scope environment diagnostics do not require a new discussion at every step.

## Privacy

Keep models, adapters, data, caches, raw logs and generations on the authorized
server. Do not request secrets in chat or export raw environments. Use approved
credentials and transfer methods. Only reviewed sanitized summaries may leave.
Follow company policies for AI tools; prompts sent to a provider may leave the
server. Never kill others' processes, force-push or change repository visibility.
