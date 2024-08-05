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
from unittest.mock import (
    patch, 
    MagicMock
)

from src.ragoon import (
    dataset_loader,
    load_datasets,
)

from ragoon._logger import Logger

logger = Logger()

@pytest.fixture
def mock_load_dataset():
    """
    Fixture to mock the load_dataset function from the datasets module.

    Yields
    ------
    mock.MagicMock
        The mocked load_dataset function.
    """
    with patch("datasets.load_dataset") as mock:
        yield mock


def test_dataset_loader_success(
    mock_load_dataset
):
    """
    Test the dataset_loader function for successful dataset loading.

    Parameters
    ----------
    mock_load_dataset : mock.MagicMock
        Mocked load_dataset function.

    Asserts
    -------
    None
    """
    mock_dataset = MagicMock()
    mock_load_dataset.return_value = mock_dataset

    name = "mock-dataset"
    dataset = dataset_loader(name)

    assert dataset == mock_dataset
    mock_load_dataset.assert_called_once_with(name, streaming=True, split=None)


def test_dataset_loader_failure(
    mock_load_dataset
):
    """
    Test the dataset_loader function for dataset loading failure.

    Parameters
    ----------
    mock_load_dataset : mock.MagicMock
        Mocked load_dataset function.

    Asserts
    -------
    None
    """
    mock_load_dataset.side_effect = Exception("Load failed")

    name = "mock-dataset"
    dataset = dataset_loader(name)

    assert dataset is None
    mock_load_dataset.assert_called_once_with(name, streaming=True, split=None)


def test_load_datasets_success(
    mock_load_dataset
):
    """
    Test the load_datasets function for successful loading of multiple datasets.

    Parameters
    ----------
    mock_load_dataset : mock.MagicMock
        Mocked load_dataset function.

    Asserts
    -------
    None
    """
    mock_dataset1 = MagicMock()
    mock_dataset2 = MagicMock()
    mock_load_dataset.side_effect = [mock_dataset1, mock_dataset2]

    req = ["mock-dataset1", "mock-dataset2"]
    datasets_list = load_datasets(req)

    assert datasets_list == [mock_dataset1, mock_dataset2]
    assert mock_load_dataset.call_count == 2


def test_load_datasets_partial_failure(
    mock_load_dataset
):
    """
    Test the load_datasets function for partial failure in loading multiple datasets.

    Parameters
    ----------
    mock_load_dataset : mock.MagicMock
        Mocked load_dataset function.

    Asserts
    -------
    None
    """
    mock_dataset1 = MagicMock()
    mock_load_dataset.side_effect = [mock_dataset1, Exception("Load failed")]

    req = ["mock-dataset1", "mock-dataset2"]
    datasets_list = load_datasets(req)

    assert datasets_list == [mock_dataset1]
    assert mock_load_dataset.call_count == 2