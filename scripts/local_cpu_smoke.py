"""Self-contained CPU plumbing check using generated data, never a model score."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.export_ms_swift_data import export_ms_swift_data
from skill_annealing.refund_decision.data_generator import export_mvp_datasets, generate_samples
from skill_annealing.refund_decision.evaluator import evaluate_records
from skill_annealing.abcd_posttraining.rule_spec import evaluate_events


def run(config, output):
    if set(config) != {'schema_version', 'sample_count', 'seed'} or config['schema_version'] != 1:
        raise ValueError('unsupported local smoke config')
    if type(config['sample_count']) is not int or not 1 <= config['sample_count'] <= 1000:
        raise ValueError('sample_count must be between 1 and 1000')
    if type(config['seed']) is not int:
        raise ValueError('integer seed required')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('use a new output directory; existing data is never overwritten')
    output.mkdir(parents=True)
    export_mvp_datasets(output / 'synthetic', sample_count=config['sample_count'], seed=config['seed'])
    export_ms_swift_data(output / 'synthetic', output / 'messages_only')
    samples = generate_samples(config['sample_count'], config['seed'])
    metrics = evaluate_records([{**r, 'predicted_output': r['target_output']} for r in samples])
    if metrics['raw_decision_accuracy'] != 1.0 or metrics['schema_valid_rate'] != 1.0:
        raise AssertionError('synthetic oracle plumbing check failed')
    events = [
        {'kind': 'fact', 'key': 'full_name_present', 'value': True, 'position': 0},
        {'kind': 'action', 'action': 'pull-up-account', 'position': 1},
        {'kind': 'fact', 'key': 'price_reason', 'value': 'yesterday', 'position': 2},
    ]
    abcd = evaluate_events('purchase_dispute/bad_price_yesterday', events)
    if abcd['action'] != 'record-reason' or abcd['independently_certified'] is not False:
        raise AssertionError('ABCD development rule fixture failed')
    inventory = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob('*.jsonl'))}
    receipt = {'schema_version': 1, 'status': 'local_cpu_smoke_pass', 'config': config,
               'synthetic_samples': len(samples), 'generated_files': inventory,
               'oracle_roundtrip_only': True, 'abcd_synthetic_fixture_pass': True,
               'model_loaded': False, 'gpu_used': False, 'real_dataset_read': False,
               'training_started': False, 'scientific_result': False}
    with (output / 'smoke.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/local_cpu.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run(json.loads(args.config.read_text(encoding='utf-8')), args.output)
    print(json.dumps({'status': receipt['status'], 'synthetic_samples': receipt['synthetic_samples'],
                      'scientific_result': False}))


if __name__ == '__main__':
    main()
