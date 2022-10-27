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

"""
Reads data from a given input.

.. deprecated:: 1.2.0
    :mod:`input.live` and :mod:`input.recorded` are deprecated.
    They have been replaced by the :mod:`input.visual` node.
"""

import sys
from typing import TYPE_CHECKING

from peekingduck.utils.lazy_module import ImportStructure, _LazyModule

_import_structure: ImportStructure = {
    "visual": [],
}

if TYPE_CHECKING:
    from . import visual
else:
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, __spec__
    )
