"""GPU-side callback; imported only by explicitly executed SWIFT training."""
import hashlib
import json
import math
import os
from pathlib import Path

from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback


def parameter_hash(model):
    h = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            raise ValueError("non-LoRA trainable parameter")
        tensor = parameter.detach().float().cpu().contiguous()
        h.update(name.encode() + str(tuple(tensor.shape)).encode() + tensor.numpy().tobytes())
        count += parameter.numel()
    if count == 0:
        raise ValueError("no trainable LoRA weights")
    return h.hexdigest(), count


class RuleExposureAudit(TrainerCallback):
    def __init__(self):
        if os.environ.get("RULE_EXPOSURE_EXECUTE") != "1":
            raise ValueError("explicit execution required")
        self.job = json.loads(Path(os.environ["RULE_EXPOSURE_JOB"]).read_text(encoding="utf-8"))
        self.path = Path(self.job["stage_dir"]) / "training_audit.json"
        self.audit = {"status": "started", "steps": []}

    def save(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.audit, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step != 0:
            raise ValueError("start from fresh base, not a trained checkpoint")
        initial, count = parameter_hash(kwargs["model"])
        if self.job.get("expected_init") and initial != self.job["expected_init"]:
            raise ValueError("LoRA initialization differs across arms")
        self.audit.update(initial_sha256=initial, trainable_parameters=count)
        self.save()

    def on_step_end(self, args, state, control, **kwargs):
        self.audit["steps"].append(int(state.global_step))
        self.save()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
            raise ValueError("nonfinite training loss")

    def on_train_end(self, args, state, control, **kwargs):
        expected = self.job["config"]["train_cases"] * 4 // self.job["config"]["gradient_accumulation_steps"]
        final, count = parameter_hash(kwargs["model"])
        if (state.global_step != expected or self.audit["steps"] != list(range(1, expected + 1))
                or count != self.audit["trainable_parameters"] or final == self.audit["initial_sha256"]):
            raise ValueError("incomplete/unchanged training endpoint")
        self.audit.update(status="completed", final_sha256=final)
        self.save()


callbacks_map["rule_exposure_audit"] = RuleExposureAudit
