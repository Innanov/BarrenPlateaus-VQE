"""Saturation depth d_sat: where a gradient-variance curve stops falling.

A random circuit's gradient variance falls with depth until the circuit is
random enough (an approximate 2-design), then stays at a flat level whose height
sets the barren plateau. d_sat is the depth where that happens. It decides the
VQE depths: the depth-4 runs sit before the plateau, d_sat at its start and
2 * d_sat well inside it. One d_sat per molecule is taken from the standard
curve. The other methods are reported relative to it (delta_d).

Rule: the flat level is the mean of the curve over the deepest depths (from
flat_from on). d_sat is the first depth from which the curve stays within a
relative tolerance (default 20%) of that level. A tolerance is used rather than
the curve's statistical uncertainty, because with five seeds of 100 samples the
uncertainty is only 2% to 7%, so a slow drift of a few percent after the decay
has ended would push d_sat far too deep (H2's flat standard curve would get
d_sat 11 instead of 4). The tolerance asks whether the decay has practically
finished. d_sat compares the curve with its own flat level, so it does not
depend on the cost's scale.

When the curve only settles inside the window that defines the flat level
(d_sat >= flat_from), it may still be falling, and d_sat is reported as not
reached (None).
"""

from __future__ import annotations

import numpy as np


def saturation_depth(
    depths, values, tolerance: float = 0.2, flat_from: int = 15
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
