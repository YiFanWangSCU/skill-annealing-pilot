"""Package an explicit text-file allowlist; never enumerate private work data.

This is a local snapshot, not a Git commit or a publication audit certificate.
No networking, credentials, model code, GPU work, or training is invoked.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


FORBIDDEN_PARTS = {'.git', '.venv', 'tmp', 'data', 'experiments', 'logs', 'secrets',
                   'upload_bundle', '__pycache__'}
ALLOWED_SUFFIXES = {'.py', '.md', '.json', '.toml', '.txt'}


def validate_name(name):
    if not isinstance(name, str):
        raise ValueError('file names must be strings')
    path = PurePosixPath(name)
    if not name or path.is_absolute() or str(path) != name or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('nonportable or escaping file name')
    if any(part.casefold() in FORBIDDEN_PARTS or part.startswith('.') for part in path.parts):
        raise ValueError('private or runtime directory excluded')
    if path.suffix not in ALLOWED_SUFFIXES or re.search(r'(?i)(password|passward|credentials|id_rsa|id_ed25519|\.env)', path.name):
        raise ValueError('sensitive or unsupported file name')
    return path


def scan_text(raw):
    text = raw.decode('utf-8-sig')
    patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'\bsk-[A-Za-z0-9_-]{24,}', r'\bgh[pousr]_[A-Za-z0-9]{24,}',
                r'(?i)\b[a-z]:[\\/](?:users|作业|projects)[\\/]',
                r'(?i)/(?:root|home)/[A-Za-z0-9_.-]+/',
                r'https?://[^\s/@:]+:[^\s/@]+@',
                r'(?i)["\x27]?(?:password|api_key|access_token)["\x27]?\s*[:=]\s*["\x27][^"\x27\r\n]{8,}["\x27]']
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError('possible secret or private machine path; inspect locally, no value printed')


def collect(root, names):
    root = Path(root).resolve(strict=True)
    if not names or len(names) > 200 or len({n.casefold() for n in names}) != len(names):
        raise ValueError('empty, duplicate or oversized allowlist')
    payload = {}
    for name in sorted(names):
        parts = validate_name(name).parts
        path = root
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ValueError('symlink in source path')
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('allowlisted file missing or outside root: ' + name)
        if path.stat().st_size > 512 * 1024:
            raise ValueError('unexpectedly large source file: ' + name)
        raw = path.read_bytes()
        scan_text(raw)
        payload[name] = raw
    if sum(map(len, payload.values())) > 8 * 1024 * 1024:
        raise ValueError('bundle exceeds bounded source scope')
    return payload


def git_provenance(root, names):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
    # A standalone extracted snapshot must not inherit an enclosing repo's HEAD.
    if not (root / '.git').exists():
        return {'head': None, 'source': 'standalone_files', 'uncommitted_files_included': None}
    try:
        head = git('rev-parse', 'HEAD')
        status = git('status', '--porcelain', '--untracked-files=all', '--', *names)
        return {'head': head, 'source': 'working_tree_snapshot', 'uncommitted_files_included': bool(status),
                'selected_path_status': status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {'head': None, 'source': 'git_metadata_unavailable', 'uncommitted_files_included': None}


def build(root, config_path, output):
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if output.exists() or output.is_symlink() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError('fresh archive and manifest paths required')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if set(config) != {'schema_version', 'scope', 'files'} or config['schema_version'] != 1:
        raise ValueError('invalid bundle configuration')
    payload = collect(root, config['files'])
    if 'bundle_manifest.json' in payload:
        raise ValueError('reserved manifest name')
    manifest = {'schema_version': 1, 'scope': config['scope'], 'provenance': git_provenance(root, config['files']),
                'files': {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                          for name, raw in payload.items()},
                'contains_real_datasets': False, 'contains_model_weights': False,
                'training_launcher_included': 'scripts/server_smoke.py' in payload,
                'secret_pattern_scan': 'passed_not_a_security_guarantee'}
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted({**payload, 'bundle_manifest.json': manifest_raw}.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    with output.with_suffix('.manifest.json').open('xb') as handle:
        handle.write(manifest_raw)
    return {'files': len(payload), 'archive_bytes': output.stat().st_size,
            'archive_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'provenance': manifest['provenance']}


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / 'bundle_manifest.json').read_text(encoding='utf-8'))
    payload = collect(root, list(manifest['files']))
    for name, raw in payload.items():
        expected = manifest['files'][name]
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('snapshot content mismatch: ' + name)
    return {'status': 'listed_snapshot_files_verified', 'files': len(payload), 'extra_local_files_checked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--config', type=Path, default=Path('configs/local_bundle.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify and args.output is None:
        parser.error('--output is required when building')
    print(json.dumps(verify(args.root) if args.verify else build(args.root, args.config, args.output), ensure_ascii=True))


if __name__ == '__main__':
    main()
