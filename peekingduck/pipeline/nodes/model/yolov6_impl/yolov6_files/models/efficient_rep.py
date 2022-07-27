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

"""Efficient Rep.

Modifications:
- Hardcode RepVGGBlock
"""

from typing import List, Tuple

import torch
from torch import nn

from peekingduck.pipeline.nodes.model.yolov6_impl.yolov6_files.layers.common import (
    RepBlock,
    RepVGGBlock,
    SimSPPF,
)


class EfficientRep(nn.Module):
    """EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """

    def __init__(  # pylint: disable=invalid-name
        self,
        in_channels: int = 3,
        channels_list: List[int] = None,
        num_repeats: List[int] = None,
    ) -> None:
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
        )

        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=RepVGGBlock,
            ),
        )

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=RepVGGBlock,
            ),
        )

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=RepVGGBlock,
            ),
        )

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=RepVGGBlock,
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input images.

        Returns:
            (Tuple[torch.Tensor, ...]):
        """
        outputs = []
        inputs = self.stem(inputs)
        inputs = self.ERBlock_2(inputs)
        inputs = self.ERBlock_3(inputs)
        outputs.append(inputs)
        inputs = self.ERBlock_4(inputs)
        outputs.append(inputs)
        inputs = self.ERBlock_5(inputs)
        outputs.append(inputs)

        return tuple(outputs)
