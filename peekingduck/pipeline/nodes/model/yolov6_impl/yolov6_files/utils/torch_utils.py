# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch utility functions.

Modifications:
- Moved initialize_weights() to YOLOv6 to resolve circular imports
"""

from typing import no_type_check

import torch
import torch.nn as nn

from peekingduck.pipeline.nodes.model.yolov6_impl.yolov6_files.layers.common import Conv
from peekingduck.pipeline.nodes.model.yolov6_impl.yolov6_files.models.yolo import YOLOv6


@no_type_check
def fuse_conv_and_bn(conv: nn.Conv2d, batch_norm: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse convolution and batchnorm layers
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(
        batch_norm.weight.div(torch.sqrt(batch_norm.eps + batch_norm.running_var))
    )
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = batch_norm.bias - batch_norm.weight.mul(batch_norm.running_mean).div(
        torch.sqrt(batch_norm.running_var + batch_norm.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model: YOLOv6) -> YOLOv6:
    """Fuses the batch normalization layers in `Conv` modules."""
    for module in model.modules():
        if isinstance(module, Conv) and hasattr(module, "bn"):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)  # update conv
            delattr(module, "bn")  # remove batchnorm
            module.forward = module.forward_fuse  # type: ignore
    return model
