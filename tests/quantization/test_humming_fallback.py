# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for humming's fallback to unquantized linear layers.

When a humming-quantized layer meets an unquantized (fp16/bf16/fp32)
checkpoint weight, HummingLinearMethod swaps its own class to
UnquantizedLinearMethod in place. The swap must also run
UnquantizedLinearMethod.__init__, which sets ``_gemm_impl``; without it,
``process_weights_after_loading`` (and ``apply``) crash with
``AttributeError: 'UnquantizedLinearMethod' object has no attribute
'_gemm_impl'`` (vllm-project/vllm#68194).
"""

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.humming import (
    HummingLayerQuantizationConfig,
    HummingLinearMethod,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.import_utils import has_humming

pytestmark = pytest.mark.skipif(
    not has_humming(), reason="humming is not installed"
)


class _FakeLinearLayer(torch.nn.Module):
    """Minimal stand-in for a LinearBase layer, carrying only the
    attributes HummingLinearMethod.create_weights and the weight loader
    read and write."""

    def __init__(self):
        super().__init__()
        self.tp_size = 1
        self.tp_rank = 0


def _make_humming_linear_method() -> HummingLinearMethod:
    import vllm.utils.humming as hm

    # int4 weight schema (GPTQ-style) with an int8 input schema, matching
    # the W4A8 setup in #68194. The layer is *not* online-quant, so an
    # fp16/bf16/fp32 checkpoint weight takes the unquantized fallback path.
    weight_schema = hm.HummingWeightSchema(
        b_dtype="int4", weight_scale_group_size=128
    )
    input_schema = hm.HummingInputSchema(a_dtype="int8")
    quant_config = HummingLayerQuantizationConfig(
        weight_schema=weight_schema, input_schema=input_schema
    )
    return HummingLinearMethod(quant_config)


def test_humming_fallback_to_unquantized_initializes_gemm_impl(dist_init):
    method = _make_humming_linear_method()
    layer = _FakeLinearLayer()
    input_size, output_size = 64, 128

    method.create_weights(
        layer,
        input_size_per_partition=input_size,
        output_partition_sizes=[output_size],
        input_size=input_size,
        output_size=output_size,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )

    # A bf16 checkpoint weight for a layer the checkpoint left unquantized
    # (e.g. a vision-tower layer, or a layer listed as 16-bit in the
    # checkpoint's extra_config).
    loaded_weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    # The fallback converts the method in place ...
    assert isinstance(method, UnquantizedLinearMethod)
    # ... and must fully initialize it, so the load-time dispatch below
    # can read _gemm_impl (vllm-project/vllm#68194).
    assert hasattr(method, "_gemm_impl")
    assert method._gemm_impl is not None

    # The loaded weight landed in the layer.
    torch.testing.assert_close(layer.weight.data, loaded_weight)

    # process_weights_after_loading is the exact crash site in #68194
    # (it compares self._gemm_impl against cuda_unquantized_gemm).
    # On CPU it takes the CPU dispatch branch instead; either way it must
    # complete without raising AttributeError for _gemm_impl.
    method.process_weights_after_loading(layer)
