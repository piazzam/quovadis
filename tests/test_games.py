import numpy as np

from quovadis_repro.quovadis import all_coalitions, exact_banzhaf, exact_shapley


def additive_values(weights):
    coalitions = all_coalitions(len(weights))
    return coalitions @ np.asarray(weights, dtype=np.float64), coalitions


def test_exact_values_on_additive_game():
    weights = np.asarray([0.25, -0.4, 1.2, 0.1])
    values, coalitions = additive_values(weights)
    np.testing.assert_allclose(exact_banzhaf(values, coalitions), weights, atol=1e-12)
    np.testing.assert_allclose(exact_shapley(values, coalitions), weights, atol=1e-12)
