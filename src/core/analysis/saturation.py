"""Saturation depth d_sat: where a gradient-variance curve stops falling.

A random circuit's gradient variance falls with depth until the circuit is
random enough (an approximate 2-design), then stays at a flat level whose height
sets the barren plateau. d_sat is the depth where that happens. It decides the
VQE depths: the depth-4 runs sit before the plateau, d_sat at its start and
2 * d_sat well inside it. One d_sat per molecule is taken from the standard
curve, the base level, and sets the VQE depths of every method.

Rule: the flat level is the mean of the curve over the deepest depths (from
flat_from on). d_sat is the first depth from which the curve stays within a
relative tolerance (TOLERANCE, 15%) of that level. A tolerance is used rather
than the curve's statistical uncertainty, because with five seeds of 100 samples
the uncertainty is only 1% to 5%, so a slow drift of a few percent after the
decay has ended would push d_sat far too deep (H2's flat standard curve would get
d_sat 11 instead of 4 at 5%). The tolerance asks whether the decay has
practically finished. d_sat compares the curve with its own flat level, so it
does not depend on the cost's scale.

15% rather than 20%: HeH+ (tapered, 6 qubits) has a standard curve that dips
18% below its flat level at depth 3 and climbs back by depth 5. At 20% that dip
counted as flat and gave d_sat 1. At 15% it gives 5, and the standard d_sat of
H2, H4, H2O, LiH and BeH2 stay as they were (4, 4, 6, 7, 9). At 10% H4 and H2O
would move to 9, closer to the noise.

When the curve only settles inside the window that defines the flat level
(d_sat >= flat_from), it may still be falling, and d_sat is reported as not
reached (None).
"""

from __future__ import annotations

import numpy as np

TOLERANCE = 0.15
"""Default relative distance from the flat level that counts as flat."""


def saturation_depth(
    depths, values, tolerance: float = TOLERANCE, flat_from: int = 15
) -> tuple[float, int | None]:
    """Return the flat level and d_sat of one gradient-variance curve.

    Args:
        depths: Circuit depths, in any order.
        values: The curve's value at each depth (mean gradient variance).
        tolerance: Relative distance from the flat level that counts as flat.
        flat_from: First depth of the window whose mean is the flat level.

    Returns:
        (flat_level, d_sat), with d_sat None when not reached before flat_from.

    Raises:
        ValueError: If no depth is at or beyond flat_from.
    """
    order = np.argsort(depths)
    d = np.asarray(depths)[order]
    v = np.asarray(values, dtype=float)[order]
    window = d >= flat_from
    if not window.any():
        raise ValueError(f"no depth at or beyond flat_from={flat_from}")
    flat = float(v[window].mean())
    inside = np.abs(v / flat - 1) <= tolerance
    for i in range(len(d)):
        if inside[i:].all():
            return flat, int(d[i]) if d[i] < flat_from else None
    return flat, None
