"""Regression tests on the aggregate artifacts exported by Google Colab.

These tests do not independently reproduce the model's ROC curves or verify patient IDs.
"""
from pathlib import Path
import csv
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def read_csv(name):
    with (RESULTS / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_four_complete_evaluations_and_test_denominators():
    rows = read_csv('metrics.csv')
    assert {r['run_id'] for r in rows} == {
        'linear_probing_400', 'lora_400',
        'linear_probing_1200', 'lora_1200',
    }
    assert len(rows) == 4
    for r in rows:
        tn, fp, fn, tp = (int(r[k]) for k in ('tn', 'fp', 'fn', 'tp'))
        assert (tn + fp, fn + tp) == (234, 390)
        assert tn + fp + fn + tp == 624
        assert float(r['sensitivity']) == pytest.approx(tp / (tp + fn))
        assert float(r['specificity']) == pytest.approx(tn / (tn + fp))
        assert float(r['balanced_accuracy']) == pytest.approx(
            (float(r['sensitivity']) + float(r['specificity'])) / 2
        )
        assert 0 <= float(r['auroc']) <= 1


def test_nested_training_sets_and_version_metadata():
    cfg = json.loads((RESULTS / 'run_config.json').read_text())
    small = cfg['train_subset_400']
    medium = cfg['train_subset_1200']
    assert len(small) == len(set(small)) == 400
    assert len(medium) == len(set(medium)) == 1200
    assert set(small) <= set(medium)
    assert min(medium) >= 0 and max(medium) < 4708
    assert cfg['seed'] == 42
    assert cfg['checkpoint'] == 'google/vit-base-patch16-224-in21k'
    assert cfg['test_evaluation'] == 'one final pass'


def test_run_summaries_and_validation_histories_agree():
    summaries = read_csv('run_summaries.csv')
    history = read_csv('validation_history.csv')
    results = read_csv('metrics.csv')
    assert len(summaries) == 4
    for s in summaries:
        run = s['run_id']
        epochs = sorted((r for r in history if r['run_id'] == run),
                        key=lambda r: int(r['epoch']))
        assert len(epochs) == int(s['epochs_run'])
        assert [int(x['epoch']) for x in epochs] == list(range(1, len(epochs) + 1))
        best = max(epochs, key=lambda x: float(x['validation_auroc']))
        assert float(s['best_validation_auroc']) == pytest.approx(
            float(best['validation_auroc']))
        assert int(s['best_epoch']) == int(best['epoch'])
        row = next(r for r in results if r['run_id'] == run)
        assert float(row['best_validation_auroc']) == pytest.approx(float(s['best_validation_auroc']))
        assert int(row['optimizer_updates']) == int(s['optimizer_updates'])
        assert int(row['trainable_parameters']) == int(s['trainable_parameters'])


def test_primary_auc_observations():
    rows = {r['run_id']: r for r in read_csv('metrics.csv')}
    auc = lambda name: float(rows[name]['auroc'])
    assert auc('lora_400') > auc('linear_probing_400')
    assert auc('lora_1200') > auc('linear_probing_1200')
    assert auc('lora_400') > auc('linear_probing_1200')
    assert float(rows['lora_400']['balanced_accuracy']) < float(
        rows['linear_probing_400']['balanced_accuracy'])
