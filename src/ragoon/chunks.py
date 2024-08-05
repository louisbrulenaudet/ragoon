# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import uuid

from concurrent.futures import (
    ThreadPoolExecutor, 
    as_completed
)

from dataclasses import dataclass
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

from datasets import (
    Dataset, 
    DatasetDict
)

from tqdm import tqdm
from transformers import AutoTokenizer

from ragoon._logger import Logger

logger = Logger()


@dataclass
class ChunkMetadata:
    """
    Metadata for a text chunk within a dataset.

    Attributes
    ----------
    uuid : str
        The UUID of the original text.

    chunk_uuid : str
        The UUID of the chunked text.

    chunk_number : str
        The identifier of the chunk indicating its order and total number of chunks.
    """
    uuid: str
    chunk_uuid: str
    chunk_number: str


class DatasetChunker:
    """
    A class to chunk text data within a dataset for processing with embeddings models.

    This class splits large texts into smaller chunks based on a specified maximum token limit,
    while maintaining an overlap between chunks to preserve context.

    Parameters
    ----------
    dataset : Union[datasets.Dataset, datasets.DatasetDict]
        The dataset to be chunked. It can be either a `Dataset` or a `DatasetDict`.

    max_tokens : int
        The maximum number of tokens allowed in each chunk.

    overlap_percentage : float
        The percentage of tokens to overlap between consecutive chunks.

    column : str
        The name of the column containing the text to be chunked.

    model_name : str, optional
        The name of the tokenizer model to use (default is "bert-base-uncased").

    uuid_column : Optional[str], optional
        The name of the column containing UUIDs for the texts. If not provided, new UUIDs will be generated.

    separators : List[str], optional
        List of separators used to split the text.

    space_after_splitters : Optional[List[str]], optional
        List of separators that require a space after splitting (default is None).

    Examples
    --------
    >>> from datasets import load_dataset
    >>> dataset = load_dataset("louisbrulenaudet/dac6-instruct")
    >>> chunker = DatasetChunker(
    ...     dataset['train'],
    ...     max_tokens=512,
    ...     overlap_percentage=0.5,
    ...     column="document",
    ...     model_name="intfloat/multilingual-e5-large",
    ...     separators=["\n", ".", "!", "?"]
    ... )
    >>> dataset_chunked = chunker.chunk_dataset()
    >>> dataset_chunked.to_list()[:3]
    [{'text': 'This is a chunked text.'}, {'text': 'This is another chunked text.'}, ...]
    """
    def __init__(
        self,
        dataset: Union[Dataset, DatasetDict],
        max_tokens: int,
        overlap_percentage: float,
        column: str,
        model_name: str = "bert-base-uncased",
        uuid_column: Optional[str] = None,
        separators: List[str] = [".", "\n"],
        space_after_splitters: Optional[List[str]] = None
    ) -> None:
        self._validate_inputs(
            dataset, 
            max_tokens, 
            overlap_percentage, 
            column, 
            model_name, 
            uuid_column, 
            separators
        )

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.overlap_percentage = overlap_percentage
        self.column = column
        self.uuid_column = uuid_column
        self.separators = separators
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.space_after_splitters = space_after_splitters or ['.', '!', '?']
        self.splitter_pattern = re.compile(f"({'|'.join(re.escape(s) for s in self.separators)})")


    @staticmethod
    def _validate_inputs(
        dataset, 
        max_tokens, 
        overlap_percentage, 
        column, 
        model_name, 
        uuid_column, 
        separators
    ):
        """
        Validates the inputs for the DatasetChunker class.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict]
            The dataset to be chunked.

        max_tokens : int
            The maximum number of tokens allowed in each chunk.

        overlap_percentage : float
            The percentage of tokens to overlap between consecutive chunks.

        column : str
            The name of the column containing the text to be chunked.

        model_name : str
            The name of the tokenizer model to use.

        uuid_column : Optional[str]
            The name of the column containing UUIDs for the texts.

        separators : List[str]
            List of separators used to split the text.

        Raises
        ------
        AssertionError
            If any of the inputs are not valid.
        """
        assert isinstance(dataset, (Dataset, DatasetDict)), "dataset must be of type Dataset or DatasetDict"
        assert isinstance(max_tokens, int) and max_tokens > 0, "max_tokens must be a positive integer"
        assert isinstance(overlap_percentage, float) and 0 <= overlap_percentage < 1, "overlap_percentage must be a float between 0 and 1"
        assert isinstance(column, str) and column, "column must be a non-empty string"
        assert isinstance(model_name, str) and model_name, "model_name must be a non-empty string"
        if uuid_column is not None:
            assert isinstance(uuid_column, str) and uuid_column, "uuid_column must be a non-empty string if provided"
        assert isinstance(separators, list) and all(isinstance(s, str) for s in separators), "separators must be a list of strings"


    def split_text(
        self, 
        text: str
    ) -> List[str]:
        """
        Splits a text into segments based on the specified separators.

        Parameters
        ----------
        text : str
            The text to be split.

        Returns
        -------
        List[str]
            A list of text segments.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> chunker.split_text("This is a sentence. This is another one.")
        ['This is a sentence', '.', ' This is another one', '.']
        """
        return [seg for seg in self.splitter_pattern.split(text) if seg]


    def create_chunks(
        self, 
        text: str
    ) -> List[str]:
        """
        Creates text chunks from a given text based on the maximum tokens limit.

        Parameters
        ----------
        text : str
            The text to be chunked.

        Returns
        -------
        List[str]
            A list of text chunks.

        Raises
        ------
        ValueError
            If the text cannot be chunked properly.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> text = "This is a very long text that needs to be chunked."
        >>> chunks = chunker.create_chunks(text)
        >>> len(chunks)
        2
        """
        self.text = text  # Store the original text
        segments = self.split_text(text)
        chunks, current_chunk = [], []
        current_tokens, overlap_tokens = 0, int(self.max_tokens * self.overlap_percentage)

        for i, segment in enumerate(segments):
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                if segment in self.space_after_splitters and not next_segment.startswith(' '):
                    segment += ' '

            segment_tokens = len(self.tokenizer.encode(segment))

            if current_tokens + segment_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(self.finalize_chunk("".join(current_chunk), is_last=(i == len(segments) - 1)))

                overlap_segments = self._get_overlap_segments(current_chunk, overlap_tokens)
                current_chunk = overlap_segments + [segment]
                current_tokens = sum(len(self.tokenizer.encode(seg)) for seg in current_chunk)
            
            else:
                current_chunk.append(segment)
                current_tokens += segment_tokens

        if current_chunk:
            chunks.append(self.finalize_chunk("".join(current_chunk), is_last=True))

        return chunks


    def finalize_chunk(
        self, 
        chunk_text: str, 
        is_last: bool
    ) -> str:
        """
        Finalizes the chunk text by adjusting leading/trailing separators.

        Parameters
        ----------
        chunk_text : str
            The chunk text to be finalized.

        is_last : bool
            Indicates whether this is the last chunk.

        Returns
        -------
        str
            The finalized chunk text.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> chunk = " This is a chunk."
        >>> chunker.finalize_chunk(chunk, is_last=True)
        'This is a chunk.'
        """
        chunk_text = chunk_text.lstrip()
        
        if chunk_text and chunk_text[0] in self.separators and not self.text.startswith(chunk_text[0]):
            chunk_text = chunk_text[1:].lstrip()
        
        if is_last and self.text[-1] in self.separators and not chunk_text.endswith(self.text[-1]):
            chunk_text = chunk_text.rstrip() + self.text[-1]
        
        elif not is_last and chunk_text[-1] not in self.separators:
            chunk_text = chunk_text.rstrip() + '.'
        
        return chunk_text


    def _get_overlap_segments(
        self, 
        segments: List[str], 
        overlap_tokens: int
    ) -> List[str]:
        """
        Retrieves segments to be used as the overlapping part of a chunk.

        Parameters
        ----------
        segments : List[str]
            The list of text segments.

        overlap_tokens : int
            The number of tokens to overlap between chunks.

        Returns
        -------
        List[str]
            A list of segments that will be included in the overlap.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> segments = ["Segment 1", "Segment 2", "Segment 3"]
        >>> chunker._get_overlap_segments(segments, 10)
        ['Segment 2']
        """
        overlap_segments, total_tokens = [], 0

        for segment in reversed(segments):
            segment_tokens = len(self.tokenizer.encode(segment))
            
            if total_tokens + segment_tokens > overlap_tokens:
                break
            
            overlap_segments.insert(0, segment)
            total_tokens += segment_tokens
        
        return overlap_segments

    def chunk_dataset(
        self
    ) -> Union[Dataset, DatasetDict]:
        """
        Chunks the entire dataset into smaller segments.

        Returns
        -------
        Union[Dataset, DatasetDict]
            The chunked dataset, with each entry split into smaller chunks.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> chunked_dataset = chunker.chunk_dataset()
        >>> len(chunked_dataset)
        1000
        """
        if isinstance(self.dataset, DatasetDict):
            return self._chunk_dataset_dict()
        
        else:
            return self._chunk_single_dataset(self.dataset)


    def _chunk_dataset_dict(self) -> DatasetDict:
        """
        Chunks each split in a `DatasetDict`.

        Returns
        -------
        DatasetDict
            A new `DatasetDict` with each split chunked.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset_dict, 512, 0.1, 'text')
        >>> chunked_dataset_dict = chunker._chunk_dataset_dict()
        >>> len(chunked_dataset_dict)
        3
        """
        new_splits = {
            split_name: self._chunk_single_dataset(split) for split_name, split in self.dataset.items()
        }
        logger.info("Chunking completed successfully for DatasetDict.")
        
        return DatasetDict(new_splits)


    def _chunk_single_dataset(
        self, 
        dataset: Dataset
    ) -> Dataset:
        """
        Chunks a single `Dataset` into smaller segments.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be chunked.

        Returns
        -------
        Dataset
            A new `Dataset` with the text data chunked.

        Raises
        ------
        KeyError
            If a row is missing the specified column.

        Exception
            If an unexpected error occurs during processing.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> chunked_dataset = chunker._chunk_single_dataset(dataset)
        >>> len(chunked_dataset)
        500
        """
        new_data = []
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_row, row) for row in dataset]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Splitting rows"):
                try:
                    new_data.extend(future.result())
                
                except KeyError as e:
                    logger.error(f"KeyError while processing row: {e}. Row data: {future.result()}")
                
                except Exception as e:
                    pass
                    # logger.error(f"Unexpected error while processing row: {e}. Row data: {future.result()}")
        
        logger.info("Chunking completed successfully for Dataset.")
        
        return Dataset.from_list(new_data)

    
    def _process_row(
        self, 
        row: dict
    ) -> List[dict]:
        """
        Processes a single row of the dataset to create chunks.

        Parameters
        ----------
        row : dict
            The row of the dataset to be processed.

        Returns
        -------
        List[dict]
            A list of rows representing the chunks.

        Raises
        ------
        KeyError
            If the specified column is not found in the row.

        Exception
            If an unexpected error occurs while processing the row.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> row = {'text': "This is a long text that needs to be chunked."}
        >>> chunked_rows = chunker._process_row(row)
        >>> len(chunked_rows)
        2
        """
        try:
            text = row[self.column]
            chunks = self.create_chunks(text)
            
            return self._create_chunk_rows(row, chunks)
        
        except KeyError as e:
            logger.error(f"KeyError: {e}. Row: {row}")
            raise
        
        except Exception as e:
            # logger.error(f"Error processing row: {e}. Row: {row}")
            # raise
            pass

    def _create_chunk_rows(
        self, 
        original_row: dict, 
        chunks: List[str]
    ) -> List[dict]:
        """
        Creates rows for each chunk with metadata.

        Parameters
        ----------
        original_row : dict
            The original row of the dataset.

        chunks : List[str]
            The list of text chunks.

        Returns
        -------
        List[dict]
            A list of new rows with chunk metadata.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> original_row = {'text': "This is a chunked text."}
        >>> chunks = ["This is a chunked", " text."]
        >>> chunk_rows = chunker._create_chunk_rows(original_row, chunks)
        >>> len(chunk_rows)
        2
        """
        original_uuid = original_row.get(self.uuid_column, str(uuid.uuid4()))
        
        return [self._create_chunk_row(original_row, chunk, ChunkMetadata(
                uuid=original_uuid,
                chunk_uuid=str(uuid.uuid4()),
                chunk_number=f"{i+1:05d}-of-{len(chunks):05d}"
            )) for i, chunk in enumerate(chunks)]


    def _create_chunk_row(
        self, 
        original_row: dict, 
        chunk: str, 
        metadata: ChunkMetadata
    ) -> dict:
        """
        Creates a new row for a chunk with metadata.

        Parameters
        ----------
        original_row : dict
            The original row of the dataset.

        chunk : str
            The text chunk.

        metadata : ChunkMetadata
            The metadata for the chunk.

        Returns
        -------
        dict
            The new row with the chunk and its metadata.

        Examples
        --------
        >>> chunker = DatasetChunker(dataset, 512, 0.1, 'text')
        >>> original_row = {'text': "This is a chunked text."}
        >>> chunk = "This is a chunked text."
        >>> metadata = ChunkMetadata(uuid="123", chunk_uuid="456", chunk_number="00001-of-00002")
        >>> new_row = chunker._create_chunk_row(original_row, chunk, metadata)
        >>> new_row
        {'text': 'This is a chunked text.', 'uuid': '123', 'chunk_uuid': '456', 'chunk_number': '00001-of-00002'}
        """
        new_row = {
            key: value for key, value in original_row.items() if key != self.column
        }

        new_row.update(
            {
                self.column: chunk,
                "uuid": metadata.uuid,
                "chunk_uuid": metadata.chunk_uuid,
                "chunk_number": metadata.chunk_number
            }
        )

        return new_row