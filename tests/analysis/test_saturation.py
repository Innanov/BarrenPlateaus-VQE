"""Tests for the saturation depth d_sat (src/core/analysis/saturation.py).

Run with: pytest tests/analysis/test_saturation.py
"""

import pytest

from src.core.analysis.saturation import saturation_depth

DEPTHS = list(range(1, 21))


def test_decay_then_flat():
    """A decay onto a flat level saturates where it stays within 15% of it."""
    values = [1.0 + 100 * 0.5**d for d in DEPTHS]  # within 15% from d = 10
    flat, d_sat = saturation_depth(DEPTHS, values)
    assert flat == pytest.approx(1.0, rel=1e-2)
    assert d_sat == 10


def test_dip_below_the_tolerance_is_not_flat():
    """A dip of 18% that climbs back (HeH+ tapered) saturates after the dip."""
    values = [1.0, 0.89, 0.82, 0.84, 0.87, 0.9] + [1.0] * 14
    assert saturation_depth(DEPTHS, values)[1] == 5
    assert saturation_depth(DEPTHS, values, tolerance=0.2)[1] == 1


def test_small_drift_does_not_delay_saturation():
    """A slow drift of a few percent after the decay does not delay d_sat."""
    values = [5.0, 3.0] + [1.0 + 0.01 * d for d in DEPTHS[2:]]
    assert saturation_depth(DEPTHS, values)[1] == 3


def test_still_falling_is_not_reached():
    """A curve still falling through the flat window has no d_sat."""
    values = [100 * 0.7**d for d in DEPTHS]
    assert saturation_depth(DEPTHS, values)[1] is None


def test_order_of_depths_does_not_matter():
    values = [1.0 + 100 * 0.5**d for d in DEPTHS]
    assert saturation_depth(DEPTHS[::-1], values[::-1]) == saturation_depth(
        DEPTHS, values
    )


def test_needs_depths_in_the_flat_window():
    with pytest.raises(ValueError):
        saturation_depth([1, 2, 3], [3.0, 2.0, 1.0], flat_from=15)


def test_pauli_weight_of_the_first_stage_costs():
    """local_global's cost and adiabatic's first stage weigh 1/(4n) and n."""
    from src.core.analysis import cost_observable
    from src.core.analysis.normalization import pauli_weight
    from src.utils.helpers import ising_system

    system = ising_system(4)
    local = pauli_weight(cost_observable(system, "local_cost"))
    reference = pauli_weight(cost_observable(system, "hf_reference"))
    assert local == pytest.approx(1 / 16)
    assert reference == pytest.approx(4)
