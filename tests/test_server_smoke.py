import json
from pathlib import Path

import pytest

from scripts.server_smoke import plan, smoke_environment, verify_training
from scripts import server_smoke


def model_fixture(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    return model


def test_dry_run_has_no_output_or_training(tmp_path):
    model = model_fixture(tmp_path)
    output = tmp_path / 'run'
    result = plan(model, output, '0')
    assert not output.exists()
    assert result['scientific_result'] is False
    command = result['command']
    for flag, value in [('--max_steps', '2'), ('--push_to_hub', 'false'),
                        ('--truncation_strategy', 'delete'), ('--strict', 'true'),
                        ('--lazy_tokenize', 'true'), ('--report_to', 'none')]:
        assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize('gpu', ['', '-1', '0,1', '0; echo bad'])
def test_reject_gpu_selection(tmp_path, gpu):
    with pytest.raises(ValueError):
        plan(model_fixture(tmp_path), tmp_path / 'run', gpu)


def test_no_overwrite_or_model_writes(tmp_path):
    model = model_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        plan(model, model, '0')
    with pytest.raises(ValueError):
        plan(model, model / 'output', '0')


def test_reject_other_model(tmp_path):
    model = model_fixture(tmp_path)
    (model / 'config.json').write_text('{"model_type":"qwen3"}')
    with pytest.raises(ValueError):
        plan(model, tmp_path / 'run', '0')


def test_environment_does_not_mutate_parent(monkeypatch):
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'NPROC_PER_NODE', 'NNODES'):
        monkeypatch.delenv(key, raising=False)
    env = smoke_environment('2')
    assert env['CUDA_VISIBLE_DEVICES'] == '2'
    assert env['HF_HUB_OFFLINE'] == '1'
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError):
        smoke_environment('2')


def test_success_needs_step_receipt_and_adapter(tmp_path):
    with pytest.raises(ValueError):
        verify_training(tmp_path)
    checkpoint = tmp_path / 'training' / 'checkpoint-2'
    checkpoint.mkdir(parents=True)
    state = checkpoint / 'trainer_state.json'
    state.write_text('{"global_step":2,"max_steps":2}')
    with pytest.raises(ValueError):
        verify_training(tmp_path)
    (checkpoint / 'adapter_config.json').write_text('{}')
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture-not-real-model')
    assert verify_training(tmp_path)['optimizer_steps'] == 2
    state.write_text('{"global_step":1,"max_steps":2}')
    with pytest.raises(ValueError):
        verify_training(tmp_path)


@pytest.mark.parametrize('fail', [False, True])
def test_execute_uses_mocked_child_only(tmp_path, monkeypatch, fail):
    import subprocess
    from types import SimpleNamespace
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'NPROC_PER_NODE', 'NNODES'):
        monkeypatch.delenv(key, raising=False)
    # Replace this module's OS reference, never mutate the process OS identity.
    monkeypatch.setattr(server_smoke, 'os', SimpleNamespace(name='posix', environ={}))
    monkeypatch.setattr(server_smoke, 'version', lambda name:
                        {'ms-swift': '4.1.1', 'transformers': '5.5.4'}.get(name, 'fixture'))
    output = tmp_path / 'run'
    prepared = plan(model_fixture(tmp_path), output, '0')
    def fake_child(command, **kwargs):
        assert kwargs['shell'] is False
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert Path(kwargs['env']['HF_HOME']).is_relative_to(output)
        if fail:
            raise subprocess.CalledProcessError(1, command)
        checkpoint = output / 'training/checkpoint-2'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'trainer_state.json').write_text('{"global_step":2,"max_steps":2}')
        (checkpoint / 'adapter_config.json').write_text('{}')
        (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture')
    monkeypatch.setattr(server_smoke.subprocess, 'run', fake_child)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            server_smoke.execute(prepared, output)
    else:
        assert server_smoke.execute(prepared, output)['status'] == 'two_update_training_smoke_pass'
    receipt = json.loads((output / 'server_smoke.json').read_text())
    assert receipt['scientific_result'] is False
    assert (receipt['status'] == 'failed_or_interrupted_inspect_local_log') == fail
