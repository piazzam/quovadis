import numpy as np

from quovadis_repro.templates import CONFIGURATIONS, generate_partition


def test_configuration_sizes():
    assert len(CONFIGURATIONS["A"]) == 17
    assert len(CONFIGURATIONS["B"]) == 10
    assert len(CONFIGURATIONS["C"]) == 8


def test_every_template_covers_all_pixels_and_players():
    for spec in CONFIGURATIONS["A"]:
        partition = generate_partition(32, 32, spec.players, spec.mode)
        assert partition.shape == (32, 32)
        assert np.array_equal(np.unique(partition), np.arange(spec.players))
