"""Opt-in, two-update Qwen3.5 LoRA plumbing check; never a paper experiment."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.local_cpu_smoke import run as cpu_smoke


def plan(model, output, gpu):
    model, output = Path(model).resolve(), Path(output).resolve()
    if not re.fullmatch(r'0|[1-9][0-9]*', gpu):
        raise ValueError('select one numeric GPU index')
    if not model.is_dir() or not (model / 'config.json').is_file():
        raise ValueError('a complete local model directory is required; no Hub download')
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('this smoke supports Qwen3.5 dense only; do not rename a different model')
    if output.exists() or output.is_symlink():
        raise FileExistsError('use a fresh output directory')
    if output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError('model and output directories must be disjoint')
    command = [sys.executable, '-m', 'swift.cli.sft',
        '--model', str(model), '--model_type', 'qwen3_5', '--template', 'qwen3_5',
        '--tuner_type', 'lora', '--dataset', str(output / 'cpu/messages_only/full_skill_sft.jsonl'),
        '--split_dataset_ratio', '0', '--max_steps', '2', '--num_train_epochs', '1',
        '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '1',
        '--learning_rate', '1e-4', '--lora_rank', '8', '--lora_alpha', '16',
        '--target_modules', 'all-linear', '--freeze_llm', 'false',
        '--freeze_vit', 'true', '--freeze_aligner', 'true', '--torch_dtype', 'bfloat16',
        '--attn_impl', 'sdpa', '--max_length', '4096', '--truncation_strategy', 'delete',
        '--strict', 'true', '--lazy_tokenize', 'true', '--packing', 'false', '--padding_free', 'false',
        '--add_non_thinking_prefix', 'true', '--use_logits_to_keep', 'true',
        '--gradient_checkpointing', 'true', '--dataset_num_proc', '1',
        '--dataloader_num_workers', '0', '--load_from_cache_file', 'false',
        '--save_strategy', 'steps', '--save_steps', '2', '--save_total_limit', '1',
        '--logging_steps', '1', '--report_to', 'none', '--push_to_hub', 'false',
        '--seed', '42', '--data_seed', '42', '--check_model', 'false',
        '--output_dir', str(output / 'training')]
    return {'status': 'dry_run_no_training', 'scientific_result': False,
            'max_optimizer_steps': 2, 'gpu': gpu, 'command': command}


def smoke_environment(gpu):
    env = os.environ.copy()
    # Refuse inherited distributed launches rather than touch other processes.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'NPROC_PER_NODE', 'NNODES'):
        if key in env:
            raise ValueError('run from a plain single-process shell, not a distributed launcher')
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', MODELSCOPE_OFFLINE='1', WANDB_MODE='disabled',
               PYTHONUNBUFFERED='1', PYTHONHASHSEED='42', DO_NOT_TRACK='1')
    return env


def verify_training(output):
    checkpoints = list((Path(output) / 'training').rglob('checkpoint-2/trainer_state.json'))
    if len(checkpoints) != 1:
        raise ValueError('expected exactly one two-update checkpoint receipt')
    state_path = checkpoints[0]
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('global_step') != 2 or state.get('max_steps') != 2:
        raise ValueError('optimizer step receipt does not match smoke budget')
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        path = state_path.parent / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('missing saved adapter')
    return {'status': 'two_update_training_smoke_pass', 'optimizer_steps': 2,
            'scientific_result': False, 'inference_verified': False}


def execute(prepared, output):
    if os.name != 'posix':
        raise ValueError('GPU execution is intended for Linux; dry-run works on all platforms')
    env = smoke_environment(prepared['gpu'])
    packages = {name: version(name) for name in
                ('ms-swift', 'torch', 'transformers', 'peft', 'accelerate', 'safetensors')}
    if packages['ms-swift'] != '4.1.1' or packages['transformers'] != '5.5.4':
        raise ValueError('use the documented isolated smoke environment versions')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env.update(HF_HOME=str(output / 'cache/hf'), MODELSCOPE_CACHE=str(output / 'cache/modelscope'),
               XDG_CACHE_HOME=str(output / 'cache/xdg'), TORCH_HOME=str(output / 'cache/torch'))
    cpu_smoke({'schema_version': 1, 'sample_count': 24, 'seed': 42}, output / 'cpu')
    model = Path(prepared['command'][prepared['command'].index('--model') + 1])
    receipt = {'status': 'started', 'scientific_result': False, 'packages': packages,
               'model_config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
               'max_optimizer_steps': 2}
    # Runtime receipts/logs stay local; command paths are not published.
    receipt_path = output / 'server_smoke.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    try:
        with (output / 'training.log').open('x', encoding='utf-8') as log:
            subprocess.run(prepared['command'], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, shell=False)
        receipt.update(verify_training(output))
    except (Exception, KeyboardInterrupt):
        receipt['status'] = 'failed_or_interrupted_inspect_local_log'
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='one allocated GPU index; never inferred')
    parser.add_argument('--execute', action='store_true', help='actually run two optimizer updates')
    args = parser.parse_args()
    prepared = plan(args.model, args.output, args.gpu)
    print(json.dumps(execute(prepared, args.output) if args.execute else prepared, indent=2))


if __name__ == '__main__':
    main()
