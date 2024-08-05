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

from datasets import load_dataset, Dataset

from src.ragoon import (
    ChunkMetadata,
    DatasetChunker
)


@pytest.fixture(scope="module")
def dataset():
    """
    Fixture to load the dataset from Hugging Face.

    Returns
    -------
    datasets.DatasetDict
        The loaded dataset.
    """
    return load_dataset("louisbrulenaudet/dacc6-instrut")


@pytest.fixture(scope="module")
def chunker(dataset):
    """
    Fixture to initialize the DatasetChunker with example parameters.

    Parameters
    ----------
    dataset : datasets.DatasetDict
        The loaded dataset fixture.

    Returns
    -------
    DatasetChunker
        The initialized DatasetChunker.
    """
    return DatasetChunker(
        dataset=dataset['train'],
        max_tokens=128,
        overlap_percentage=0.1,
        column="text",
        model_name="bert-base-uncased"
    )


def test_split_text(
    chunker
):
    """
    Test the split_text method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    text = "This is a sentence. This is another one."
    expected_output = ['This is a sentence', '.', ' This is another one', '.']
    
    assert chunker.split_text(text) == expected_output


def test_create_chunks(
    chunker
):
    """
    Test the create_chunks method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    text = "This is a very long text that needs to be chunked. " * 50
    chunks = chunker.create_chunks(text)
    
    assert len(chunks) > 1  # Ensure that the text is split into multiple chunks
    assert all(len(chunker.tokenizer.encode(chunk)) <= chunker.max_tokens for chunk in chunks)


def test_finalize_chunk(
    chunker
):
    """
    Test the finalize_chunk method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    chunk = " This is a chunk."
    assert chunker.finalize_chunk(chunk, is_last=True) == "This is a chunk."
    chunk = " This is another chunk"
    assert chunker.finalize_chunk(chunk, is_last=False) == "This is another chunk."


def test_chunk_dataset(
    chunker
):
    """
    Test the chunk_dataset method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    chunked_dataset = chunker.chunk_dataset()
    
    assert isinstance(chunked_dataset, Dataset)
    assert len(chunked_dataset) > len(chunker.dataset)  # Ensure dataset is chunked into more rows


def test_process_row(
    chunker
):
    """
    Test the _process_row method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    row = {
        'text': "This is a long text that needs to be chunked. " * 10
    }
    chunked_rows = chunker._process_row(row)
    
    assert len(chunked_rows) > 1  # Ensure multiple rows are created
    assert all('chunk_uuid' in chunk for chunk in chunked_rows)
    assert all('chunk_number' in chunk for chunk in chunked_rows)


def test_create_chunk_rows(
    chunker
):
    """
    Test the _create_chunk_rows method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    original_row = {'text': "This is a chunked text."}
    chunks = ["This is a chunked", " text."]
    chunked_rows = chunker._create_chunk_rows(original_row, chunks)
    
    assert len(chunked_rows) == 2
    assert all(isinstance(chunk, dict) for chunk in chunked_rows)


def test_create_chunk_row(
    chunker
):
    """
    Test the _create_chunk_row method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    original_row = {'text': "This is a chunked text."}
    chunk = "This is a chunked text."
    metadata = ChunkMetadata(
        uuid="123", 
        chunk_uuid="456", 
        chunk_number="00001-of-00002"
    )
    new_row = chunker._create_chunk_row(original_row, chunk, metadata)
    
    assert new_row['text'] == "This is a chunked text."
    assert new_row['uuid'] == "123"
    assert new_row['chunk_uuid'] == "456"
    assert new_row['chunk_number'] == "00001-of-00002"


def test_get_overlap_segments(
    chunker
):
    """
    Test the _get_overlap_segments method of DatasetChunker.

    Parameters
    ----------
    chunker : DatasetChunker
        The initialized DatasetChunker fixture.
    """
    segments = ["Segment 1", "Segment 2", "Segment 3"]
    overlap_tokens = 10
    overlap_segments = chunker._get_overlap_segments(segments, overlap_tokens)

    assert isinstance(overlap_segments, list)