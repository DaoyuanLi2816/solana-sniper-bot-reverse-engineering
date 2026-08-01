import numpy as np
import pandas as pd

from solana_sniper.baseline import (
    NEGATIVE_SAMPLE_WEIGHT,
    _fixed_threshold_metrics,
    _population_weights,
)


def test_population_weights_expand_sampled_negatives() -> None:
    labels = pd.Series([1, 0, 0, 1])
    assert _population_weights(labels).tolist() == [1.0, NEGATIVE_SAMPLE_WEIGHT, 25.0, 1.0]


def test_weighted_operating_point_reports_conservative_precision() -> None:
    labels = pd.Series([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.8, 0.1])
    unweighted = _fixed_threshold_metrics(labels, probabilities, 0.5, np.ones(4))
    weighted = _fixed_threshold_metrics(labels, probabilities, 0.5, _population_weights(labels))
    assert weighted["precision"] < unweighted["precision"]
