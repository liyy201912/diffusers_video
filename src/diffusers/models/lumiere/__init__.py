from ...utils import is_torch_available


if is_torch_available():
    from .blocks import ConvInflationBlock, AttentionInflationBlock, TemporalDownsample, TemporalUpsample, PostModuleHookWrapper
