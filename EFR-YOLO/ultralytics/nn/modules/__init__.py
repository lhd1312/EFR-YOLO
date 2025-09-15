# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxslim {f} {f} && open {f}')  # pip install onnxslim
    ```
"""

from .block import (
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    HGBlock,
    HGStem,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose,  Segment, WorldDetect, v10Detect
from .transformer import (
    LayerNorm2d,
    MLPBlock,
    TransformerBlock,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "Bottleneck",
    "BottleneckCSP",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "OBB",
    "WorldDetect",
    "v10Detect",

)

