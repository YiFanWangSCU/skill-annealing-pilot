import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.build_local_bundle import build, collect, scan_text, validate_name, verify
from scripts.local_cpu_smoke import run


@pytest.mark.parametrize('name', ['../secret.py', '/secret.py', 'tmp/x.py', 'data/x.json',
    '.env', 'x/password.txt', 'a\\b.py', 'C:/x.py', 'a/../b.py', 'model.safetensors'])
def test_unsafe_paths_are_excluded(name):
    with pytest.raises(ValueError):
        validate_name(name)


def test_pattern_scan_fails_without_exposing_value():
    value = ('s' + 'k-' + 'x' * 32).encode()
    with pytest.raises(ValueError) as error:
        scan_text(value)
    assert value.decode() not in str(error.value)


def test_duplicate_and_missing_sources_fail(tmp_path):
    with pytest.raises(ValueError):
        collect(tmp_path, ['A.py', 'a.py'])
    with pytest.raises(ValueError):
        collect(tmp_path, ['absent.py'])


def test_bundle_is_deterministic_and_detects_tampering(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'one.py').write_text('x = 1\n')
    config = source / 'selection.json'
    config.write_text(json.dumps({'schema_version': 1, 'scope': 'fixture', 'files': ['one.py']}))
    a, b = tmp_path / 'a.zip', tmp_path / 'b.zip'
    assert build(source, config, a)['archive_sha256'] == build(source, config, b)['archive_sha256']
    unpacked = tmp_path / 'unpacked'
    with zipfile.ZipFile(a) as archive:
        archive.extractall(unpacked)  # Only the just-built, path-validated fixture archive.
    assert verify(unpacked)['files'] == 1
    (unpacked / 'one.py').write_text('x = 2\n')
    with pytest.raises(ValueError):
        verify(unpacked)
    with pytest.raises(FileExistsError):
        build(source, config, a)


def test_smoke_uses_generated_data_and_is_repeatable(tmp_path):
    config = {'schema_version': 1, 'sample_count': 24, 'seed': 42}
    a, b = run(config, tmp_path / 'a'), run(config, tmp_path / 'b')
    assert a == b
    assert a['status'] == 'local_cpu_smoke_pass'
    assert a['scientific_result'] is False and a['model_loaded'] is False
    assert len(a['generated_files']) == 8
    with pytest.raises(FileExistsError):
        run(config, tmp_path / 'a')


def test_smoke_config_has_no_implicit_server_fields(tmp_path):
    with pytest.raises(ValueError):
        run({'schema_version': 1, 'sample_count': 24, 'seed': 42, 'model': 'anything'}, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_default_allowlist_has_all_required_local_sources():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/local_bundle.json').read_text())
    payload = collect(root, config['files'])
    assert 'skill_annealing/abcd_posttraining/rule_spec.py' in payload
    assert not any(name.startswith(('data/', 'tmp/', 'experiments/')) for name in payload)
