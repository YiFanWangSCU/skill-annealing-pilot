"""Deterministic paired policy data. No original corpus or hidden split is read."""
import hashlib
import itertools
import json
from pathlib import Path
import random
from collections import Counter

ARMS = ("full_only", "short_only", "whole_mix", "component_mix")
VIEWS = ("full", "short")
DECISIONS = ("manual_review", "request_more_evidence", "allow_refund",
             "allow_return_refund", "deny")
BASE = {"return_days": 7, "review_amount": 1000, "customized_allowed": False}
TRAIN_POLICIES = [
    BASE, {**BASE, "return_days": 14}, {**BASE, "review_amount": 1500},
    {**BASE, "customized_allowed": True},
]
EVAL_POLICIES = [
    ("base", BASE),
    ("unseen_combination", {"return_days": 14, "review_amount": 1500, "customized_allowed": True}),
    ("unseen_values", {"return_days": 10, "review_amount": 1250, "customized_allowed": False}),
    ("unseen_values", {"return_days": 21, "review_amount": 750, "customized_allowed": True}),
]
MODULES = (
    "优先级1：若订单事实冲突，或订单金额大于等于当前 review_amount，输出 manual_review。",
    "优先级2：未触发优先级1时，若签收天数未知，输出 request_more_evidence。",
    "优先级3：未触发前两项时，若有质量问题，有证据输出 allow_refund，无证据输出 request_more_evidence。",
    "优先级4：无质量问题时，虚拟商品拒绝；定制商品只有 customized_allowed=true 才能进入普通退货判断。",
    "优先级5：其余情况仅当签收天数小于等于 return_days、未使用且可二次销售时输出 allow_return_refund，否则输出 deny。",
)
TASK = ("这是模拟业务规则任务，不是真实退款政策。根据订单事实和当前有效配置，"
        '仅输出一个 JSON 对象 {"decision":"标签"}，不得增加其他字段。'
        "标签可选：" + " / ".join(DECISIONS) + "。"
        "规则编号仅用于说明判断优先级；省略的流程已在训练中学习，配置从不省略。")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def decide(facts, policy):
    """Ordered, explicitly specified simulator; NOT the legacy refund gold."""
    if facts["conflict"] or facts["amount"] >= policy["review_amount"]:
        return "manual_review"
    if facts["days"] is None:
        return "request_more_evidence"
    if facts["quality"]:
        return "allow_refund" if facts["evidence"] else "request_more_evidence"
    if facts["kind"] == "virtual" or (facts["kind"] == "customized" and not policy["customized_allowed"]):
        return "deny"
    if facts["days"] <= policy["return_days"] and not facts["used"] and facts["resellable"]:
        return "allow_return_refund"
    return "deny"


def messages(facts, policy, mask, answer=False):
    if len(mask) != len(MODULES) or any(type(x) is not bool for x in mask):
        raise ValueError("one boolean exposure per stable module required")
    system = TASK + "\n当前有效配置（全部字段有效）：" + canonical(policy)
    rules = [module for module, keep in zip(MODULES, mask) if keep]
    if rules:
        system += "\n" + "\n".join(rules)
    rows = [{"role": "system", "content": system},
            {"role": "user", "content": "请处理此订单的退款请求。订单事实：" + canonical(facts)}]
    if answer:
        rows.append({"role": "assistant", "content": canonical({"decision": decide(facts, policy)})})
    return rows


def validate_config(config):
    required = {"protocol", "claim_id", "seed", "train_cases", "eval_cases", "arms",
                "gradient_accumulation_steps", "max_length", "max_new_tokens", "lora_rank",
                "lora_alpha", "learning_rate", "stage_timeout_seconds",
                "campaign_timeout_seconds", "bootstrap_resamples"}
    if set(config) != required or config["protocol"] != "rule_exposure_pilot_v1":
        raise ValueError("unsupported pilot config")
    if config["claim_id"] != "RULE-EXPOSURE-ADAPTATION" or config["arms"] != list(ARMS):
        raise ValueError("all four registered arms required")
    for name, low, high in [
        ("seed", 0, 2**31-1), ("train_cases", 10, 500), ("eval_cases", 10, 200),
        ("gradient_accumulation_steps", 1, 32), ("max_length", 512, 4096),
        ("max_new_tokens", 16, 128), ("lora_rank", 1, 32), ("lora_alpha", 1, 128),
        ("stage_timeout_seconds", 30, 7200), ("campaign_timeout_seconds", 30, 28800),
        ("bootstrap_resamples", 100, 5000),
    ]:
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError("invalid bounded config field: " + name)
    if type(config["learning_rate"]) not in (float, int) or not 0 < config["learning_rate"] <= 0.001:
        raise ValueError("invalid learning rate")
    if config["train_cases"] * 4 % config["gradient_accumulation_steps"]:
        raise ValueError("training records must be divisible by effective batch")
    return config


def case_splits(config):
    keys = ("days", "amount", "kind", "used", "resellable", "quality", "evidence", "conflict")
    values = ([None, 0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 15, 21, 22],
              [100, 200, 300, 400, 500, 600, 700, 750, 800, 900, 1000, 1100, 1250, 1400, 1500, 1600],
              ["ordinary", "customized", "virtual"], *([[False, True]] * 5))
    buckets = {key: [] for key in DECISIONS}
    for values_i in itertools.product(*values):
        facts = dict(zip(keys, values_i))
        buckets[decide(facts, BASE)].append(facts)
    rng = random.Random(config["seed"])
    for bucket in buckets.values():
        rng.shuffle(bucket)
    # Oracle-stratified for an informative pilot; not a natural prevalence estimate.
    selected = []
    for index in range(config["train_cases"] + config["eval_cases"]):
        label = DECISIONS[index % len(DECISIONS)]
        selected.append(buckets[label].pop())
    train, evaluation = selected[:config["train_cases"]], selected[config["train_cases"]:]
    assert not {digest(x) for x in train} & {digest(x) for x in evaluation}
    return train, evaluation


def datasets(config):
    validate_config(config)
    train, evaluation = case_splits(config)
    paired = [(facts, policy) for facts in train for policy in TRAIN_POLICIES]
    random.Random(config["seed"] + 1).shuffle(paired)
    n = len(paired)
    whole = [True] * (n // 2) + [False] * (n // 2)
    random.Random(config["seed"] + 2).shuffle(whole)
    components = []
    for index in range(len(MODULES)):
        keep = [True] * (n // 2) + [False] * (n // 2)
        random.Random(config["seed"] + 10 + index).shuffle(keep)
        components.append(keep)
    rows, sidecars = {}, {}
    for arm in ARMS:
        rows[arm], sidecars[arm] = [], []
        for index, (facts, policy) in enumerate(paired):
            mask = ([True] * 5 if arm == "full_only" else [False] * 5 if arm == "short_only"
                    else [whole[index]] * 5 if arm == "whole_mix"
                    else [col[index] for col in components])
            chat = messages(facts, policy, mask, answer=True)
            rows[arm].append({"messages": chat})
            sidecars[arm].append({"case_id": digest(facts), "policy_id": digest(policy),
                                 "target": decide(facts, policy), "mask": mask,
                                 "messages_sha256": digest(chat)})
    prompts, reference = [], []
    for facts in evaluation:
        case_id = digest(facts)
        for index, (category, policy) in enumerate(EVAL_POLICIES):
            for view in VIEWS:
                rid = f"{case_id[:20]}:p{index}:{view}"
                prompts.append({"id": rid, "messages": messages(facts, policy, [view == "full"] * 5)})
                reference.append({"id": rid, "case_id": case_id, "policy_index": index,
                                  "policy_category": category, "view": view,
                                  "target": decide(facts, policy), "base_target": decide(facts, BASE),
                                  "affected": decide(facts, policy) != decide(facts, BASE)})
    return rows, sidecars, prompts, reference, train


def prepare(output, config):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("prepare requires a fresh data directory")
    rows, sidecars, prompts, reference, train = datasets(config)
    output.mkdir(parents=True)
    for arm in ARMS:
        write_rows(output / (arm + ".jsonl"), rows[arm])
        write_rows(output / (arm + ".audit.jsonl"), sidecars[arm])
    write_rows(output / "eval_prompts.jsonl", prompts)
    write_rows(output / "eval_reference.jsonl", reference)
    manifest = {
        "protocol": config["protocol"], "config": config, "config_sha256": digest(config),
        "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))},
        "train_records_per_arm": len(rows[ARMS[0]]),
        "optimizer_steps": len(rows[ARMS[0]]) // config["gradient_accumulation_steps"],
        "eval_prompts_per_endpoint": len(prompts), "endpoints": ["base", *ARMS],
        "train_policy_ids": [digest(x) for x in TRAIN_POLICIES],
        "eval_policies": [{"category": kind, "policy": policy} for kind, policy in EVAL_POLICIES],
        "train_base_label_counts": dict(Counter(decide(f, BASE) for f in train)),
        "affected_eval_rows": sum(row["affected"] for row in reference),
        "real_data": False, "historical_panel_reused": False, "exploratory": True,
        "structured_facts_oracle_available": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_data(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = validate_config(manifest["config"])
    if manifest["config_sha256"] != digest(config):
        raise ValueError("config hash mismatch")
    rows, audits, prompts, refs, _ = datasets(config)
    expected = {"eval_prompts.jsonl": prompts, "eval_reference.jsonl": refs}
    expected.update({arm + ".jsonl": rows[arm] for arm in ARMS})
    expected.update({arm + ".audit.jsonl": audits[arm] for arm in ARMS})
    if set(manifest["files"]) != set(expected):
        raise ValueError("unexpected dataset inventory")
    for name, items in expected.items():
        if file_hash(output / name) != manifest["files"][name] or read_rows(output / name) != items:
            raise ValueError("dataset differs from deterministic protocol: " + name)
    if manifest["optimizer_steps"] != len(rows[ARMS[0]]) // config["gradient_accumulation_steps"]:
        raise ValueError("step budget mismatch")
    if manifest["eval_prompts_per_endpoint"] != len(prompts) or manifest["endpoints"] != ["base", *ARMS]:
        raise ValueError("evaluation budget mismatch")
    return manifest
