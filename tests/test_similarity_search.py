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

import numpy as np

from src.ragoon import SimilaritySearch


@pytest.fixture
def similarity_search_instance():
    """
    Fixture to create an instance of `SimilaritySearch`.

    Returns
    -------
    SimilaritySearch
        An instance of `SimilaritySearch` class.
    """
    return SimilaritySearch(
        model_name="louisbrulenaudet/tsdae-lemone-mbert-base",
        device="cpu",
        ndim=768,
        metric="ip",
        dtype="i8"
    )

def test_encode(
    similarity_search_instance: SimilaritySearch
):
    """
    Test the `encode` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    Raises
    ------
    AssertionError
        If the output shape of embeddings is not as expected.
    """
    corpus = [
        "This is a test sentence.", 
        "Another sentence for encoding."
    ]
    embeddings = similarity_search_instance.encode(corpus)

    assert embeddings.shape[0] == len(corpus)
    assert embeddings.shape[1] == similarity_search_instance.ndim


def test_quantize_embeddings(
    similarity_search_instance: SimilaritySearch
):
    """
    Test the `quantize_embeddings` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    Raises
    ------
    AssertionError
        If the shape of quantized embeddings does not match the input embeddings.
    """
    corpus = [
        "This is a test sentence."
    ]
    embeddings = similarity_search_instance.encode(corpus)

    quantized_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings, 
        "int8"
    )

    assert quantized_embeddings.shape == embeddings.shape

def test_create_faiss_index(
    similarity_search_instance: SimilaritySearch
):
    """
    Test the `create_faiss_index` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    Raises
    ------
    AssertionError
        If the FAISS index is not created successfully.
    """
    corpus = [
        "This is a test sentence."
    ]
    embeddings = similarity_search_instance.encode(corpus)

    ubinary_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings, 
        "ubinary"
    )

    similarity_search_instance.create_faiss_index(ubinary_embeddings)

    assert similarity_search_instance.binary_index is not None


def test_create_usearch_index(
    similarity_search_instance
):
    """
    Test the `create_usearch_index` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    Raises
    ------
    AssertionError
        If the USEARCH index is not created successfully.
    """
    corpus = [
        "This is a test sentence."
    ]
    embeddings = similarity_search_instance.encode(corpus)
    
    int8_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings,
        "int8"
    )

    similarity_search_instance.create_usearch_index(int8_embeddings)

    assert similarity_search_instance.int8_index is not None


def test_load_usearch_index_view(
    similarity_search_instance, 
    tmp_path
):
    """
    Test the `load_usearch_index_view` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    tmp_path : pathlib.Path
        Temporary directory path provided by pytest.

    Raises
    ------
    AssertionError
        If the USEARCH index view is not loaded successfully.
    """
    corpus = [
        "This is a test sentence."
    ]

    embeddings = similarity_search_instance.encode(corpus)

    int8_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings, "int8"
    )
    index_path = tmp_path / "usearch_int8.index"

    similarity_search_instance.create_usearch_index(
        int8_embeddings, 
        index_path=index_path, 
        save=True
    )

    loaded_index = similarity_search_instance.load_usearch_index_view(index_path)
    
    assert loaded_index is not None


def test_load_faiss_index(
    similarity_search_instance, 
    tmp_path
):
    """
    Test the `load_faiss_index` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    tmp_path : pathlib.Path
        Temporary directory path provided by pytest.

    Raises
    ------
    AssertionError
        If the FAISS index is not loaded successfully.
    """
    corpus = [
        "This is a test sentence."
    ]

    embeddings = similarity_search_instance.encode(corpus)

    ubinary_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings, 
        "ubinary"
    )

    index_path = tmp_path / "faiss_ubinary.index"

    similarity_search_instance.create_faiss_index(
        ubinary_embeddings, 
        index_path=index_path, 
        save=True
    )

    similarity_search_instance.load_faiss_index(index_path)

    assert similarity_search_instance.binary_index is not None


def test_search(
    similarity_search_instance
):
    """
    Test the `search` method of `SimilaritySearch`.

    Parameters
    ----------
    similarity_search_instance : SimilaritySearch
        An instance of the `SimilaritySearch` class.

    Raises
    ------
    AssertionError
        If the search does not return any results.
    """
    corpus = [
        "This is a test sentence.", 
        "Another sentence for testing search."
    ]

    embeddings = similarity_search_instance.encode(corpus)
    ubinary_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings, 
        "ubinary"
    )

    int8_embeddings = similarity_search_instance.quantize_embeddings(
        embeddings, 
        "int8"
    )

    similarity_search_instance.create_faiss_index(ubinary_embeddings)
    similarity_search_instance.create_usearch_index(int8_embeddings)

    query = "test sentence"
    top_k_scores, top_k_indices = similarity_search_instance.search(query)
    
    assert len(top_k_scores) > 0
    assert len(top_k_indices) > 0