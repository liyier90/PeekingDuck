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

"""YOLOv6 model implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.base import (
    ThresholdCheckerMixin,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.yolov6_impl.yolov6_files.detector import Detector


class YOLOv6Model(  # pylint: disable=too-few-public-methods
    ThresholdCheckerMixin, WeightsDownloaderMixin
):
    """Validates configuration, loads YOLOv6 model, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.check_bounds(["iou_threshold", "score_threshold"], "[0, 1]")

        model_dir = self.download_weights()
        with open(model_dir / self.weights["classes_file"]) as infile:
            class_names = [line.strip() for line in infile.readlines()]

        self.detect_ids = self.config["detect"]
        self.detector = Detector(
            model_dir,
            class_names,
            self.detect_ids,
            self.config["model_type"],
            self.config["num_classes"],
            self.weights["model_file"],
            self.config["max_detections"],
            self.config["agnostic_nms"],
            self.config["fuse"],
            self.config["half"],
            self.config["multi_label"],
            self.config["input_size"],
            self.config["iou_threshold"],
            self.config["score_threshold"],
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_object_bbox_from_image(image)
