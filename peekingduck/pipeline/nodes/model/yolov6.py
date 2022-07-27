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

"""YOLOv6 custom node."""

from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.yolov6_impl import yolov6_model


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.config["detect"] = list(range(self.config["num_classes"]))
        self.model = yolov6_model.YOLOv6Model(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        bboxes, labels, scores = self.model.predict(inputs["img"])
        bboxes = np.clip(bboxes, 0, 1)
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs
