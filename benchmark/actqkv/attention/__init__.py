from .rope import RotaryEmbeddingESM

from .inf_llm import inf_llm_forward
from .qllm import qllm_forward
from .actqkv import actqkv_forward
from .actqkv_activate_only import actqkv_activate_only_forward
from .actqkv_dynamic_only import actqkv_dynamic_only_forward
from .actqkv_dev import actqkv_dev_forward
from .infinite_lm import infinite_lm_forward
from .stream_llm import stream_llm_forward
from .origin import origin_forward
from .actqkv_dev_28 import actqkv_dev_28_forward

ATTN_FORWRAD = {
    "inf-llm": inf_llm_forward,
    'qllm': qllm_forward,
    "infinite-lm": infinite_lm_forward,
    "stream-llm": stream_llm_forward,
    "origin": origin_forward,
    'actqkv_dev': actqkv_dev_forward,
    'actqkv': actqkv_forward,
    'actqkv_activate_only': actqkv_activate_only_forward,
    'actqkv_dynamic_only': actqkv_dynamic_only_forward,
    'actqkv_dev_qwen': actqkv_dev_28_forward
}

__all__ = ["RotaryEmbeddingESM", "ATTN_FORWARD"]