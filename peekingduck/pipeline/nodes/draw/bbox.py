# Copyright 2021 AI Singapore
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

"""
Draws bounding boxes over detected objects.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.draw.utils.bbox import draw_bboxes
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws bounding boxes on image.

    The ``draw.bbox`` node uses the ``bboxes`` and, optionally, the
    ``bbox_labels`` from the model predictions to draw the bbox predictions
    onto the image. For better understanding of the use case, refer to the
    :doc:`Object Counting use case </use_cases/object_counting>`.

    Inputs:
        |img|

        |bboxes|

        |bbox_labels|

    Outputs:
        |no_output|

    Configs:
        show_labels (:obj:`bool`): **default = False**. |br|
            If ``True``, shows class label, e.g. "person", above the bounding
            box
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        draw_bboxes(
            inputs["img"], inputs["bboxes"], inputs["bbox_labels"], self.show_labels
        )
        return {}
