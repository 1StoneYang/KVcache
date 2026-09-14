"""Install MM-ShiftKV-Re on top of the original MM-ShiftKV runtime."""

from __future__ import annotations

import functools
import inspect
import os

from mmshift_re.cluster import update_kv_with_recycle
from mmshift_re.context import set_visual_mask


_INSTALLED = False


def apply() -> None:
    """Patch Qwen2.5-VL loading so shiftkv_re = original ShiftKV + Recycling Bin."""
    global _INSTALLED
    if _INSTALLED:
        return

    import lmms_eval.models.qwen2_5_vl as qwen_mod

    original_init = qwen_mod.Qwen2_5_VL.__init__

    def wrapped_init(self, *args, **kwargs):
        recycle = os.getenv("METHOD", "") == "shiftkv_re"
        if recycle:
            os.environ["METHOD"] = "shiftkv"
        try:
            original_init(self, *args, **kwargs)
        finally:
            if recycle:
                os.environ["METHOD"] = "shiftkv_re"
                _install_recycle_hooks()
                print(f"Using shiftkv_re! Recycling Bin B={os.getenv('BIN_SIZE', '20')}")

    qwen_mod.Qwen2_5_VL.__init__ = wrapped_init
    _INSTALLED = True


def _install_recycle_hooks() -> None:
    import mmshift.models.qwen_model as qwen_model
    import mmshift.utils.mmshift_util as mmshift_util
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen25
    from mmshift.utils.mmshift_util import ShiftKVCluster

    ShiftKVCluster.update_kv = update_kv_with_recycle

    original_init_shiftkv = mmshift_util.init_shiftkv

    def init_shiftkv_re(self):
        method = os.getenv("METHOD", "")
        swapped = method == "shiftkv_re"
        if swapped:
            os.environ["METHOD"] = "shiftkv"
        try:
            return original_init_shiftkv(self)
        finally:
            if swapped:
                os.environ["METHOD"] = "shiftkv_re"

    mmshift_util.init_shiftkv = init_shiftkv_re
    qwen_model.init_shiftkv = init_shiftkv_re

    original_forward = qwen25.Qwen2_5_VLForConditionalGeneration.forward

    @functools.wraps(original_forward)
    def wrapped_forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if (
            input_ids is not None
            and hasattr(input_ids, "shape")
            and input_ids.shape[-1] > 1
        ):
            image_id = getattr(self.config, "image_token_id", None)
            if image_id is not None:
                set_visual_mask(input_ids[0] == image_id)
        return original_forward(self, *args, **kwargs)

    wrapped_forward.__signature__ = inspect.signature(original_forward)
    qwen25.Qwen2_5_VLForConditionalGeneration.forward = wrapped_forward
