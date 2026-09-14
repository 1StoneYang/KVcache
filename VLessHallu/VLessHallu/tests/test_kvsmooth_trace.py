from __future__ import annotations

import pytest

from vlesshallu.transformers_runtime import (
    _finish_kvsmooth_summary,
    _update_kvsmooth_summary,
)


def test_kvsmooth_trace_is_reduced_to_bounded_statistics() -> None:
    summary = {
        "step_count": 0,
        "layer_update_count": 0,
        "coefficient_sum": 0.0,
        "coefficient_min": None,
        "coefficient_max": None,
        "entropy_sum": 0.0,
        "entropy_min": None,
        "entropy_max": None,
    }
    _update_kvsmooth_summary(
        summary,
        [
            {"lambda": 0.5, "entropy": 1.0},
            {"lambda": 0.9, "entropy": 3.0},
        ],
    )
    result = _finish_kvsmooth_summary(summary)
    assert result["step_count"] == 1
    assert result["layer_update_count"] == 2
    assert result["coefficient_mean"] == pytest.approx(0.7)
    assert result["coefficient_min"] == 0.5
    assert result["coefficient_max"] == 0.9
    assert result["entropy_mean"] == 2.0
    assert "coefficient_sum" not in result
    assert "entropy_sum" not in result
