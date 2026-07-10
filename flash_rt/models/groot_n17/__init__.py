"""GR00T N1.7 model package.

Architecture: Qwen3-VL-2B (Cosmos-Reason2-2B) backbone with M-RoPE +
DeepStack + 4-layer vl_self_attention + 32-layer AlternateVLDiT action
head. State/action dim 132, action horizon 40.

Per the unified pipeline_<hw>.py contract:
    pipeline_thor.py      - Thor SM110 pointer-only forward functions
    pipeline_rtx.py       - RTX SM120 (re-exports dit_forward from Thor)
    pipeline_rtx_fp8.py   - RTX SM120 FP8 backbone forwards
    pipeline_rtx_fp16.py  - RTX full-FP16 reference forwards
    pipeline_rtx_sm89.py  - RTX SM89 FP8 backbone forwards

The embodiment mapping is re-exported eagerly (no heavy deps). All other
symbols (calibration, M-RoPE tables, pipeline forwards) are available via
their submodules and are listed in ``__all__`` for discoverability but
imported lazily so that ``import flash_rt.models.groot_n17`` does not
pull in torch.
"""
from __future__ import annotations

from flash_rt.models.groot_n17.embodiments import (
    EMBODIMENT_NUM_VIEWS,
    EMBODIMENT_TAG_TO_INDEX,
)

__all__ = [
    "EMBODIMENT_TAG_TO_INDEX",
    "EMBODIMENT_NUM_VIEWS",
    "calibrate_pipeline_amax",
    "amax_to_dev_scale",
    "RopeConfig",
    "build_cos_sin_tables",
    "build_position_ids_for_segments",
    "dit_forward",
]


def __getattr__(name: str):
    if name in ("calibrate_pipeline_amax", "amax_to_dev_scale"):
        from flash_rt.models.groot_n17 import calibration
        return getattr(calibration, name)
    if name in ("RopeConfig", "build_cos_sin_tables",
                "build_position_ids_for_segments"):
        from flash_rt.models.groot_n17 import mrope_table
        return getattr(mrope_table, name)
    if name == "dit_forward":
        from flash_rt.models.groot_n17 import pipeline_rtx
        return pipeline_rtx.dit_forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
