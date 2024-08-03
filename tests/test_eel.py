# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest
import shutil

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

from src.ragoon import (
    EmbeddingsVisualizer
)
from .conftest import setup_test_data


def test_embeddings_visualizer(
    setup_test_data: str
):
    """
    Test the EEL class for embedding visualization.

    This test ensures that the `EEL` class can correctly initialize with the given
    FAISS index and dataset paths, perform PCA visualization, and save the result
    as an HTML file. It checks for the existence of the output HTML file to verify
    successful visualization.

    Parameters
    ----------
    setup_test_data : str
        Path to the temporary directory containing the test data. Provided by the `setup_test_data` fixture.

    Assertions
    ----------
    assert os.path.exists("embedding_visualization.html")
        Asserts that the HTML file `embedding_visualization.html` is created by the `EEL.visualize` method.
    """
    index_path = os.path.join(setup_test_data, "faiss_cgi_ubinary.index")
    dataset_path = os.path.join(setup_test_data, "cgi.hf")

    visualizer = EmbeddingsVisualizer(
        index_path=index_path, 
        dataset_path=dataset_path
    )

    visualizer.visualize(
        method="pca",
        save_html=True,
        html_file_name="embedding_visualization.html"
    )

    assert os.path.exists("embedding_visualization.html")