import numpy as np
import pytest
from utils import (nested_stratified_indices, class_weights, binary_metrics,
                   should_extend_after_three, should_stop_extension)


def test_nested_stratified_subsets():
    y = np.array([0] * 1100 + [1] * 3900)
    result = nested_stratified_indices(y)
    assert len(result[400]) == 400 and len(result[1200]) == 1200
    assert set(result[400]).issubset(set(result[1200]))
    assert len(np.unique(result[400])) == 400
    assert result[400].tolist() == nested_stratified_indices(y)[400].tolist()


def test_weights():
    assert np.allclose(class_weights([0, 0, 0, 1]), [2 / 3, 2])
    with pytest.raises(ValueError):
        class_weights([1, 1, 1])


def test_binary_metrics_known_case():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.7])
    assert metrics['sensitivity'] == 1.0
    assert metrics['specificity'] == 0.5
    assert metrics['balanced_accuracy'] == 0.75
    assert metrics['tp'] == 2 and metrics['tn'] == 1


def test_reject_empty_invalid_predictions():
    with pytest.raises(ValueError):
        binary_metrics([], [])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.2, 1.4])


def test_preregistered_stopping_rule():
    assert should_extend_after_three([0.60, 0.64, 0.65])
    assert not should_extend_after_three([0.60, 0.65, 0.648])
    assert should_stop_extension([0.60, 0.64, 0.65, 0.6505, 0.651])
    assert not should_stop_extension([0.60, 0.64, 0.65, 0.653, 0.654])
