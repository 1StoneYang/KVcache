from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def inference_only_tests():
    with torch.inference_mode():
        yield
