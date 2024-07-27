# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import shutil
import os

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Callable,
    Optional,
    Sequence,
)


@pytest.fixture(scope="module")
def setup_test_data(
	tmp_path_factory: pytest.TempPathFactory
) -> str:
    """
    Sets up test data for pytest by copying files from a source directory to a temporary directory.

    This fixture creates a temporary directory for the duration of the test module, copies
    all files from the `data` directory located in the same directory as this script, and
    provides the path to the temporary directory to the tests. This ensures that the original
    test data remains unchanged and isolated for each test run.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary directories, provided by pytest.

    Yields
    ------
    str
        Path to the temporary directory containing the copied test data.

    Examples
    --------
    Use this fixture in your test function to get the path to the test data directory:

    >>> def test_example(setup_test_data):
    >>>     data_dir = setup_test_data
    >>>     # Your test code here
    """
    test_data_dir = tmp_path_factory.mktemp("data")
    src_data_dir = os.path.join(os.path.dirname(__file__), "data")

    def copy_item(src, dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    for item in os.listdir(src_data_dir):
        s = os.path.join(src_data_dir, item)
        d = os.path.join(test_data_dir, item)
        copy_item(s, d)

    yield str(test_data_dir)