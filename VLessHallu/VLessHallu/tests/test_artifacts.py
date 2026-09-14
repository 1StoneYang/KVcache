from __future__ import annotations

from vlesshallu.artifacts import RunStore
from vlesshallu.benchmark import _repair_pruning_ledger


def test_interrupted_pruning_append_is_repaired(tmp_path) -> None:
    store = RunStore.create(tmp_path, "abc", {"run_hash": "abc"})
    store.captions.append(
        {
            "sample_id": 7,
            "caption": "a caption",
            "pruning_events": [{"step": 2}],
        }
    )
    _repair_pruning_ledger(store)
    assert store.pruning.get(7) == {
        "sample_id": 7,
        "events": [{"step": 2}],
    }
