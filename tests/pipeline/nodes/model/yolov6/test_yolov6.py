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

from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.yolov6 import Node
from tests.conftest import PKD_DIR


@pytest.fixture
def yolov6_config():
    with open(PKD_DIR / "configs" / "model" / "yolov6.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"agnostic_nms": True, "fuse": True, "half": True},
        {"agnostic_nms": True, "fuse": True, "half": False},
        {"agnostic_nms": True, "fuse": False, "half": True},
        {"agnostic_nms": True, "fuse": False, "half": False},
        {"agnostic_nms": False, "fuse": True, "half": True},
        {"agnostic_nms": False, "fuse": True, "half": False},
        {"agnostic_nms": False, "fuse": False, "half": True},
        {"agnostic_nms": False, "fuse": False, "half": False},
    ]
)
def yolov6_matrix_config(request, yolov6_config):
    yolov6_config.update(request.param)
    return yolov6_config


@pytest.fixture(params=["yolov6n"])
def yolov6_config_cpu(request, yolov6_matrix_config):
    yolov6_matrix_config["model_type"] = request.param
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield yolov6_matrix_config


@pytest.mark.mlmodel
class TestYOLOv6:
    def test_no_human_image(self, no_human_image, yolov6_config_cpu):
        no_human_img = cv2.imread(no_human_image)
        yolov6 = Node(yolov6_config_cpu)
        output = yolov6.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, yolov6_config):
        with pytest.raises(ValueError) as excinfo:
            yolov6_config["weights"][yolov6_config["model_format"]]["model_file"][
                yolov6_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=yolov6_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)
