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

import random
import string
import sys
import textwrap
from pathlib import Path
from unittest import TestCase, mock

import pytest
import yaml
from click.testing import CliRunner

from peekingduck.cli import cli
from tests.conftest import assert_msg_in_logs

# The tests in script expect the following directory structure:
# <random name dir> (generated by tmp_dir fixture)
# \-tmp_dir (generated by tmp_project_dir fixture)
#   \-src
#     \-custom_nodes
#       +-configs
#       | \-pkd_node_type
#       \-pkd_node_type

UNIQUE_SUFFIX = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
CUSTOM_FOLDER_NAME = f"custom_nodes_{UNIQUE_SUFFIX}"

PKD_NODE_TYPE = "pkd_node_type"
PKD_NODE_NAME = "pkd_node_name"
PKD_NODE_NAME_2 = "pkd_node_name2"
PKD_NODE = f"{CUSTOM_FOLDER_NAME}.{PKD_NODE_TYPE}.{PKD_NODE_NAME}"
PKD_NODE_2 = f"{CUSTOM_FOLDER_NAME}.{PKD_NODE_TYPE}.{PKD_NODE_NAME_2}"
NODES = {"nodes": [PKD_NODE, PKD_NODE_2]}

MODULE_DIR = Path("tmp_dir") / "src"

NODE_DIR = MODULE_DIR / "custom_nodes"
CONFIG_DIR = NODE_DIR / "configs"
CUSTOM_NODE_DIR = MODULE_DIR / CUSTOM_FOLDER_NAME
CUSTOM_CONFIG_DIR = CUSTOM_NODE_DIR / "configs"
CUSTOM_TYPE_NODE_DIR = CUSTOM_NODE_DIR / PKD_NODE_TYPE
CUSTOM_TYPE_CONFIG_DIR = CUSTOM_CONFIG_DIR / PKD_NODE_TYPE

PIPELINE_PATH = Path("pipeline_config.yml")
CUSTOM_PIPELINE_PATH = Path("custom_dir") / "pipeline_config.yml"

DEFAULT_YML = dict(
    nodes=[
        {
            "input.visual": {
                "source": "https://storage.googleapis.com/peekingduck/videos/wave.mp4"
            }
        },
        "model.posenet",
        "draw.poses",
        "output.screen",
    ]
)


class MockRunner:
    """Mocks the Runner class to print out the pipeline_config content."""

    def __init__(
        self,
        pipeline_path=None,
        config_updates_cli=None,
        custom_nodes_parent_subdir=None,
        num_iter=None,
        nodes=None,
    ):
        with open(pipeline_path) as infile:
            data = yaml.safe_load(infile.read())
            print(data)

    def run(self):
        pass


def create_node_config(config_dir, node_name, config_text):
    with open(config_dir / f"{node_name}.yml", "w") as outfile:
        yaml.dump(config_text, outfile)


def create_node_python(node_dir, node_name, return_statement):
    with open(node_dir / f"{node_name}.py", "w") as outfile:
        content = textwrap.dedent(
            f"""\
            from peekingduck.pipeline.nodes.abstract_node import AbstractNode

            class Node(AbstractNode):
                def __init__(self, config, **kwargs):
                    super().__init__(config, node_path=__name__, **kwargs)

                def run(self, inputs):
                    return {{ {return_statement} }}
            """
        )
        outfile.write(content)


def create_pipeline_yaml(nodes, custom_config_path):
    with open(
        CUSTOM_PIPELINE_PATH if custom_config_path else PIPELINE_PATH, "w"
    ) as outfile:
        yaml.dump(nodes, outfile, default_flow_style=False)


def init_msg(node_name):
    return f"Initializing {node_name} node..."


def setup(custom_config_path=False):
    sys.path.append(str(MODULE_DIR))

    relative_node_dir = CUSTOM_TYPE_NODE_DIR.relative_to(CUSTOM_TYPE_NODE_DIR.parts[0])
    relative_config_dir = CUSTOM_TYPE_CONFIG_DIR.relative_to(
        CUSTOM_TYPE_CONFIG_DIR.parts[0]
    )
    relative_node_dir.mkdir(parents=True)
    relative_config_dir.mkdir(parents=True)
    if custom_config_path:
        CUSTOM_PIPELINE_PATH.parent.mkdir()
    node_config_1 = {
        "input": ["none"],
        "output": ["test_output_1"],
        "resize": {"do_resizing": False},
    }
    node_config_2 = {"input": ["test_output_1"], "output": ["pipeline_end"]}

    create_pipeline_yaml(NODES, custom_config_path)
    create_node_python(relative_node_dir, PKD_NODE_NAME, "'test_output_1': None")
    create_node_python(relative_node_dir, PKD_NODE_NAME_2, "'pipeline_end': True")
    create_node_config(relative_config_dir, PKD_NODE_NAME, node_config_1)
    create_node_config(relative_config_dir, PKD_NODE_NAME_2, node_config_2)


@pytest.fixture(name="cwd")
def fixture_cwd():
    """Making this a fixture allows it to be called after the `tmp_dir`
    fixture so we can get the proper path.
    """
    return Path.cwd()


@pytest.fixture(name="parent_dir")
def fixture_parent_dir():
    """Making this a fixture allows it to be called after the `tmp_dir`
    fixture so we can get the proper path.
    """
    return Path.cwd().parent


@pytest.mark.usefixtures("tmp_dir", "tmp_project_dir")
class TestCliCore:
    def test_init_default(self, parent_dir, cwd):
        """Checks the folders and file content for `peekingduck init` with
        default options.
        """
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(cli, ["init"])

            assert result.exit_code == 0
            assert_msg_in_logs(
                f"Creating custom nodes folder in {cwd / 'src' / 'custom_nodes'}",
                captured.records,
            )
            assert (parent_dir / NODE_DIR).exists()
            assert (parent_dir / CONFIG_DIR).exists()
            assert (cwd / PIPELINE_PATH).exists()
            with open(cwd / PIPELINE_PATH) as infile:
                config_file = yaml.safe_load(infile)
                TestCase().assertDictEqual(DEFAULT_YML, config_file)

    def test_init_custom(self, parent_dir, cwd):
        """Checks the folders and file content for
        `peekingduck init --custom_folder_name <some name>`.
        """
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(
                cli, ["init", "--custom_folder_name", CUSTOM_FOLDER_NAME]
            )
            assert result.exit_code == 0
            assert_msg_in_logs(
                f"Creating custom nodes folder in {cwd / 'src' / CUSTOM_FOLDER_NAME}",
                captured.records,
            )
            assert (parent_dir / CUSTOM_NODE_DIR).exists()
            assert (parent_dir / CUSTOM_CONFIG_DIR).exists()
            assert (cwd / PIPELINE_PATH).exists()
            with open(cwd / PIPELINE_PATH) as infile:
                TestCase().assertDictEqual(DEFAULT_YML, yaml.safe_load(infile))

    def test_run_default(self):
        setup()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(cli, ["run"])
            assert_msg_in_logs("Successfully loaded pipeline file.", captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE), captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE_2), captured.records)
            assert result.exit_code == 0

    def test_run_custom_config_path(self):
        setup(True)
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(
                cli, ["run", "--config_path", CUSTOM_PIPELINE_PATH]
            )
            assert_msg_in_logs("Successfully loaded pipeline file.", captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE), captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE_2), captured.records)
            assert result.exit_code == 0

    def test_run_custom_config(self):
        setup()
        node_name = ".".join(PKD_NODE.split(".")[1:])
        config_update_value = "'do_resizing': True"
        config_update_cli = (
            f"{{'{node_name}': {{'resize': {{ {config_update_value} }} }} }}"
        )
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            result = CliRunner().invoke(
                cli, ["run", "--node_config", config_update_cli]
            )

            assert_msg_in_logs("Successfully loaded pipeline file.", captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE), captured.records)
            assert_msg_in_logs(
                f"Config for node {node_name} is updated to: {config_update_value}",
                captured.records,
            )
            assert_msg_in_logs(init_msg(PKD_NODE_2), captured.records)
            assert result.exit_code == 0

    def test_run_num_iter(self):
        setup()
        with TestCase.assertLogs("peekingduck.cli.logger") as captured:
            n = 50  # run test for 50 iterations
            result = CliRunner().invoke(cli, ["run", "--num_iter", n])
            assert_msg_in_logs("Successfully loaded pipeline file.", captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE), captured.records)
            assert_msg_in_logs(init_msg(PKD_NODE_2), captured.records)
            assert_msg_in_logs(f"Run pipeline for {n} iterations", captured.records)
            assert result.exit_code == 0

    @mock.patch("peekingduck.commands.core.Runner", MockRunner)
    def test_verify_install(self):
        """Checks that verify install runs the basic object detection
        pipeline.
        """
        setup()
        result = CliRunner().invoke(cli, ["verify-install"])

        verification_pipeline_str = (
            "{'nodes': ["
            "{'input.visual': {'source': 'https://storage.googleapis.com/peekingduck/videos/wave.mp4'}}, "
            "'model.yolo', "
            "'draw.bbox', "
            "'output.screen'"
            "]}"
        )
        assert result.output.rstrip() == verification_pipeline_str
