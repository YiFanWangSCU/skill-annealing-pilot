import copy
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from scripts import rule_exposure as controller
from skill_annealing.rule_exposure import data, analysis, runtime


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "configs/rule_exposure_pilot.json").read_text())
    result.update(train_cases=20, eval_cases=10, bootstrap_resamples=100)
    return result


def facts(**kwargs):
    return {"days": 10, "amount": 100, "kind": "ordinary", "used": False,
            "resellable": True, "quality": False, "evidence": False, "conflict": False, **kwargs}


@pytest.mark.parametrize("f,p,expected", [
    (facts(), data.BASE, "deny"),
    (facts(), {**data.BASE, "return_days": 14}, "allow_return_refund"),
    (facts(days=7), data.BASE, "allow_return_refund"),
    (facts(days=8), data.BASE, "deny"),
    (facts(amount=1000), data.BASE, "manual_review"),
    (facts(amount=1000), {**data.BASE, "review_amount": 1500}, "deny"),
    (facts(quality=True), data.BASE, "request_more_evidence"),
    (facts(quality=True, evidence=True), data.BASE, "allow_refund"),
    (facts(quality=True, evidence=True, conflict=True), data.BASE, "manual_review"),
    (facts(days=None), data.BASE, "request_more_evidence"),
    (facts(days=None, amount=1000), data.BASE, "manual_review"),
    (facts(kind="customized", days=7), data.BASE, "deny"),
    (facts(kind="customized", days=7), {**data.BASE, "customized_allowed": True}, "allow_return_refund"),
    (facts(kind="virtual", days=7), {**data.BASE, "customized_allowed": True}, "deny"),
    (facts(days=7, used=True), data.BASE, "deny"),
    (facts(days=7, resellable=False), data.BASE, "deny"),
])
def test_oracle_boundaries(f, p, expected):
    assert data.decide(f, p) == expected


def test_pairs_masks_and_policy_information(config):
    rows, audits, prompts, references, train = data.datasets(config)
    n = config["train_cases"] * 4
    order = lambda arm: [(r["case_id"], r["policy_id"], r["target"]) for r in audits[arm]]
    for arm in data.ARMS:
        assert order(arm) == order("full_only")
        assert len(rows[arm]) == n
        for row in rows[arm]:
            assert "当前有效配置" in row["messages"][0]["content"]
            for key in data.BASE:
                assert key in row["messages"][0]["content"]
    for index in range(5):
        assert sum(x["mask"][index] for x in audits["component_mix"]) == n // 2
        assert sum(x["mask"][index] for x in audits["whole_mix"]) == n // 2
    assert any(0 < sum(x["mask"]) < 5 for x in audits["component_mix"])
    assert all(sum(x["mask"]) in (0, 5) for x in audits["whole_mix"])
    assert not {data.digest(x) for x in train} & {r["case_id"] for r in references}
    assert all(len(p["messages"]) == 2 and "target" not in p for p in prompts)
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert {data.digest(p) for p in data.TRAIN_POLICIES}.isdisjoint(
        data.digest(p) for category, p in data.EVAL_POLICIES if category != "base")


def test_repeatability_and_tamper_rejection(tmp_path, config):
    a, b = data.prepare(tmp_path / "a", config), data.prepare(tmp_path / "b", config)
    assert a == b
    assert data.verify_data(tmp_path / "a") == a
    with pytest.raises(FileExistsError):
        data.prepare(tmp_path / "a", config)
    path = tmp_path / "a/full_only.jsonl"
    with path.open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError):
        data.verify_data(tmp_path / "a")


def test_default_and_maximum_case_capacity(config):
    root = Path(__file__).resolve().parents[1]
    default = controller.load_config(root / "configs/rule_exposure_pilot.json")
    rows, _, prompts, refs, _ = data.datasets(default)
    assert len(rows["full_only"]) == 512 and len(prompts) == 384
    assert any(row["affected"] for row in refs)
    train, evaluation = data.case_splits({**config, "train_cases": 500, "eval_cases": 200})
    assert len(train) == 500 and len(evaluation) == 200


@pytest.mark.parametrize("change", [{"train_cases": True}, {"arms": ["full_only"]},
                                   {"campaign_timeout_seconds": 999999}, {"extra": 1}])
def test_bad_config(config, change):
    with pytest.raises(ValueError):
        data.validate_config({**config, **change})


@pytest.mark.parametrize("text,expected", [
    ('{"decision":"deny"}', "deny"),
    ('<think>\n\n</think>\n{"decision":"deny"}', "deny"),
    ('<think>reasoning</think>{"decision":"deny"}', None),
    ('```json\n{"decision":"deny"}\n```', None),
    ('{"decision":"deny","extra":1}', None),
    ('{"decision":"deny","decision":"allow_refund"}', None),
    ('{"decision":"invented"}', None), ('[]', None), ("oops", None),
])
def test_strict_scoring(text, expected):
    assert analysis.parse(text) == expected


def test_simulation_end_to_end_is_not_model_evidence(tmp_path, config):
    report = controller.simulate(tmp_path / "sim", config)
    assert report["status"] == "simulation_only_no_model_evidence"
    assert report["claim_promotion"] is False
    for endpoint in report["metrics"].values():
        assert endpoint["short"]["accuracy"] == 1
        assert endpoint["full"]["mean_input_tokens"] is None
    with pytest.raises(FileNotFoundError):
        analysis.summarize(tmp_path / "sim")
    with pytest.raises(ValueError, match="simulation"):
        controller.run(tmp_path / "sim", config, tmp_path / "model", "0")


def test_prediction_duplicates_and_backend_rejected():
    prompts = [{"id": "a"}, {"id": "b"}]
    rows = [{"id": "a", "text": "x", "simulated": True}] * 2
    with pytest.raises(ValueError):
        runtime.verify_predictions(rows, prompts, True)
    rows = [{"id": p["id"], "text": "x", "simulated": True} for p in prompts]
    with pytest.raises(ValueError):
        runtime.verify_predictions(rows, prompts, False)


def test_case_cluster_interval(config):
    _, _, _, refs, _ = data.datasets(config)
    a = {r["id"]: 1 for r in refs}
    b = {r["id"]: 0 for r in refs}
    result = analysis.paired_interval(refs, a, b, seed=1, resamples=100)
    assert result["mean_pp"] == 100
    assert result["cases"] == config["eval_cases"]  # Not twice the case count.
    assert result["ci95_pp"] == [100, 100]


def test_environment_and_dry_run_never_launch(tmp_path, config, monkeypatch):
    monkeypatch.setattr(controller, "launch", lambda *a: pytest.fail("must not launch"))
    report = controller.run(tmp_path / "run", config, tmp_path / "absent_model", "0")
    assert report["gpu_used"] is False and report["training_runs"] == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError):
        runtime.environment("0", tmp_path / "cache")
    with pytest.raises(ValueError):
        runtime.environment("0;bad", tmp_path / "cache")


def test_model_inventory_missing_and_escaping_shard(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_5"}')
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    with pytest.raises(ValueError):
        runtime.model_inventory(model)
    (model / "model.safetensors").write_bytes(b"fixture")
    assert runtime.model_inventory(model)["model_type"] == "qwen3_5"
    (model / "model.safetensors.index.json").write_text('{"weight_map":{"x":"../other.safetensors"}}')
    with pytest.raises(ValueError):
        runtime.model_inventory(model)


def test_training_command_equal_budget(config, tmp_path):
    for arm in data.ARMS:
        job = {"config": config, "arm": arm, "model": "fixture-model",
               "data": "fixture-data", "stage_dir": "fixture-run"}
        cmd = runtime.training_command(job, tmp_path)
        for key, value in [("--max_steps", "10"), ("--dataset_shuffle", "false"),
                           ("--train_dataloader_shuffle", "false"), ("--strict", "true"),
                           ("--enable_thinking", "false"),
                           ("--push_to_hub", "false")]:
            assert cmd[cmd.index(key) + 1] == value


def mocked_controller(monkeypatch):
    monkeypatch.setattr(controller, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(controller, "environment", lambda *a: {})
    monkeypatch.setattr(controller, "gpu_lock", lambda *a: nullcontext())
    monkeypatch.setattr(controller, "idle_gpu", lambda *a: {"name": "fixture"})


def write_fake_stage(job, *, invalid_init=False):
    stage = Path(job["stage_dir"])
    config = job["config"]
    result = {"status": "completed", "job_sha256": data.digest(job), "packages": {"fixture": "1"}}
    if job["kind"] == "preflight":
        result.update(model_stat_signature={}, model_inventory={"tree_sha256": "fixture"})
    elif job["kind"] == "train":
        steps = config["train_cases"] * 4 // config["gradient_accumulation_steps"]
        ckpt = stage / "training" / f"checkpoint-{steps}"
        ckpt.mkdir(parents=True)
        data.write_json(ckpt / "trainer_state.json", {"global_step": steps, "max_steps": steps})
        data.write_json(ckpt / "adapter_config.json", {})
        (ckpt / "adapter_model.safetensors").write_bytes(b"fixture-not-real")
        initial = "wrong" if invalid_init else "same-init"
        data.write_json(stage / "training_audit.json",
                        {"status": "completed", "initial_sha256": initial, "final_sha256": "changed",
                         "steps": list(range(1, steps + 1))})
        result.update(checkpoint=ckpt.relative_to(stage).as_posix(), initial_sha256=initial,
                      adapter_sha256=runtime.adapter_hash(ckpt))
    else:
        refs = data.read_rows(Path(job["data"]) / "eval_reference.jsonl")
        data.write_rows(stage / "predictions.jsonl", [
            {"id": r["id"], "text": data.canonical({"decision": r["target"]}), "simulated": False,
             "prompt_tokens": 100, "completion_tokens": 8} for r in refs])
        result["predictions_sha256"] = data.file_hash(stage / "predictions.jsonl")
    data.write_json(stage / "receipt.json", result)


def test_mock_campaign_resume_complete_and_tamper(tmp_path, config, monkeypatch):
    mocked_controller(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    calls = []
    def fake_launch(command, env, log, timeout):
        job = json.loads(Path(command[-1]).read_text())
        calls.append(job["kind"])
        write_fake_stage(job)
    monkeypatch.setattr(controller, "launch", fake_launch)
    work = tmp_path / "work"
    report = controller.run(work, config, model, "0", execute=True)
    assert len(calls) == 10 and calls.count("train") == 4
    assert report["status"] == "exploratory_model_results"  # Mock lives only in pytest temp.
    assert report["provenance"]["model_tree_sha256"] == "fixture"
    assert str(work) not in json.dumps(report)  # No local paths in shareable summaries.
    controller.run(work, config, model, "0", execute=True)
    assert len(calls) == 10
    bad = work / "stages/infer_base_attempt1/predictions.jsonl"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        controller.run(work, config, model, "0", execute=True)


def test_failed_stage_requires_explicit_retry_and_preserves_attempt(tmp_path, config, monkeypatch):
    mocked_controller(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    def fail(*a):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(controller, "launch", fail)
    work = tmp_path / "work"
    with pytest.raises(RuntimeError):
        controller.run(work, config, model, "0", execute=True)
    with pytest.raises(RuntimeError, match="retry-failed"):
        controller.run(work, config, model, "0", execute=True)
    def succeed(command, *args):
        write_fake_stage(json.loads(Path(command[-1]).read_text()))
    monkeypatch.setattr(controller, "launch", succeed)
    controller.run(work, config, model, "0", execute=True, retry_failed=True)
    assert (work / "stages/preflight_attempt1/job.json").exists()
    assert (work / "stages/preflight_attempt2/receipt.json").exists()


def test_mismatched_initialization_stops_before_next_arm(tmp_path, config, monkeypatch):
    mocked_controller(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    def launch(command, *args):
        job = json.loads(Path(command[-1]).read_text())
        write_fake_stage(job, invalid_init=job["arm"] == "short_only")
    monkeypatch.setattr(controller, "launch", launch)
    with pytest.raises(ValueError, match="initialization"):
        controller.run(tmp_path / "work", config, model, "0", execute=True)
    assert not (tmp_path / "work/stages/train_whole_mix_attempt1").exists()


def test_worker_preflight_uses_training_and_inference_templates(tmp_path, config, monkeypatch):
    import sys
    from scripts import rule_exposure_worker as worker
    data.prepare(tmp_path / "data", config)
    modes, seen = [], []
    class Template:
        def set_mode(self, mode):
            modes.append(mode)
        def encode(self, row):
            seen.append(row)
            return {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]}
    def get_processor(model, **kwargs):
        assert kwargs["load_model"] is False and kwargs["download_model"] is False
        return None, "fixture-processor"
    def get_template(processor, **kwargs):
        assert processor == "fixture-processor"
        assert kwargs["enable_thinking"] is False
        assert kwargs["truncation_strategy"] == "raise"
        return Template()
    monkeypatch.setitem(sys.modules, "swift", SimpleNamespace(
        get_model_processor=get_processor, get_template=get_template))
    monkeypatch.setattr(worker, "model_inventory", lambda _: {"files": {}, "tree_sha256": "fixture"})
    result = worker.preflight({"model": "fixture", "config": config, "data": str(tmp_path / "data")})
    assert modes == ["train", "transformers"]
    assert len(seen) == config["train_cases"] * 4 * 4 + config["eval_cases"] * 8
    assert result["tokenization_checked"]


def test_worker_inference_passes_messages_only(tmp_path, config, monkeypatch):
    import sys
    from scripts import rule_exposure_worker as worker
    data.prepare(tmp_path / "data", config)
    stage = tmp_path / "stage"
    stage.mkdir()
    received = []
    class Engine:
        def __init__(self, model, **kwargs):
            assert kwargs["max_batch_size"] == 1
        def infer(self, requests, request, **kwargs):
            assert request.temperature == 0 and request.max_tokens == config["max_new_tokens"]
            assert set(vars(requests[0])) == {"messages"}
            assert all(row["role"] != "assistant" for row in requests[0].messages)
            received.append(requests[0])
            return [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"decision":"deny"}'))],
                                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=6))]
    def get_processor(model, **kwargs):
        assert kwargs["download_model"] is False and kwargs["device_map"] == "cuda:0"
        return SimpleNamespace(eval=lambda: None), "fixture"
    def template(processor, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "fixture-template"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda _: None, bfloat16="fixture", cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "swift", SimpleNamespace(
        get_model_processor=get_processor, get_template=template))
    monkeypatch.setitem(sys.modules, "swift.infer_engine", SimpleNamespace(
        TransformersEngine=Engine, InferRequest=SimpleNamespace, RequestConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "swift.tuners", SimpleNamespace(Swift=None))
    result = worker.infer({"model": "fixture", "config": config,
                          "data": str(tmp_path / "data"), "stage_dir": str(stage)})
    assert result["prediction_count"] == config["eval_cases"] * 8
    assert len(received) == result["prediction_count"]


def test_summary_rejects_missing_training_stage(tmp_path, config, monkeypatch):
    mocked_controller(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(controller, "launch", lambda command, *args:
                        write_fake_stage(json.loads(Path(command[-1]).read_text())))
    work = tmp_path / "work"
    controller.run(work, config, model, "0", execute=True)
    state = json.loads((work / "campaign.json").read_text())
    del state["stages"]["train_full_only"]
    data.write_json(work / "campaign.json", state)
    with pytest.raises(ValueError, match="inventory"):
        analysis.summarize(work)
