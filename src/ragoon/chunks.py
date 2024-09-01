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

from huggingface_hub import InferenceClient
from tqdm import tqdm
from transformers import AutoTokenizer

from ragoon._logger import Logger

logger = Logger()


class SemanticTextSplitter:
    """
    A class for splitting text into semantically coherent sections using a language model.

    This class leverages the Hugging Face Inference API to generate splits in the input text,
    and then processes the result to return a list of split text sections. It is designed
    to work with various language models available through the Hugging Face platform.

    Parameters
    ----------
    model : str, optional
        The name or path of the Hugging Face model to use for text splitting.
        This should be a model capable of text generation tasks, such as GPT-based models.
        Default is 'meta-llama/Meta-Llama-3.1-70B-Instruct'.

    token : str, optional
        The Hugging Face API token for authentication. If not provided, the class will
        attempt to use the token stored in the Hugging Face CLI configuration.

    split_token : str, optional
        The token used to split the text (default is '<|split|>'). This token will be
        inserted by the model to indicate where the text should be split.

    system_prompt : str, optional
        The system prompt to use for the model. If not provided, a default prompt
        will be used, which instructs the model on how to split the text.

    max_tokens : int, optional
        The maximum number of tokens to generate in the model's response (default is 4096).
        This limit applies to the entire response, including the input prompt.

    stream : bool, optional
        Whether to stream the model's output (default is True). When True, the output
        will be printed as it's generated. When False, the output will be returned all at once.

    Attributes
    ----------
    client : InferenceClient
        The Hugging Face Inference API client used to communicate with the model.

    split_token : str
        The token used to split the text.

    system_prompt : str
        The system prompt used to instruct the model on how to split the text.

    max_tokens : int
        The maximum number of tokens to generate in the model's response.

    stream : bool
        Whether to stream the model's output.

    Methods
    -------
    completion(text: str) -> str
        Calls the language model to process the input text.

    split(text: str) -> List[str]
        Splits the input text into semantically coherent sections.

    Raises
    ------
    ValueError
        If the model name is not provided during initialization.

    RuntimeError
        If there's an error calling the Hugging Face Inference API.

    Examples
    --------
    >>> # Ensure you have set up your Hugging Face token using `huggingface-cli login`
    >>> splitter = SemanticTextSplitter(
    ...     model="meta-llama/Llama-2-70b-chat-hf",
    ...     token=api.token  # This will use your stored Hugging Face token
    ... )
    >>> text = '''
    ... The Python programming language, created by Guido van Rossum,
    ... has become one of the most popular languages in the world.
    ... Its simplicity and readability make it an excellent choice for beginners.
    ... Meanwhile, data science has emerged as a crucial field in the modern world.
    ... Python's extensive libraries, such as NumPy and Pandas, have made it
    ... a favorite among data scientists and analysts.
    ... '''
    >>> result = splitter.split(text)
    >>> for section in result:
    ...     print(f"Section: {section}\n")
    Section: The Python programming language, created by Guido van Rossum,
    has become one of the most popular languages in the world.
    Its simplicity and readability make it an excellent choice for beginners.

    Section: Meanwhile, data science has emerged as a crucial field in the modern world.
    Python's extensive libraries, such as NumPy and Pandas, have made it
    a favorite among data scientists and analysts.

    Notes
    -----
    - The quality of the text splitting depends on the capabilities of the chosen language model.
    - The system prompt plays a crucial role in guiding the model's behavior. Customizing it
      can lead to different splitting results.
    - When using streamed output, the results are printed to the console in real-time,
      which can be useful for monitoring long-running splits.
    - The split token ('<split>' by default) should be chosen carefully to avoid conflicts
      with the content of the text being split.

    See Also
    --------
    huggingface_hub.InferenceClient : The client used to interact with Hugging Face models.
    """

    def __init__(
        self,
        model: Optional[str] = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        token: Optional[str] = None,
        split_token: str = "<|split|>",
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        stream: bool = True,
    ):
        """
        Initialize the SemanticTextSplitter.

        Parameters
        ----------
        model : str, optional
            The name or path of the Hugging Face model to use for text splitting.
            Default is 'meta-llama/Meta-Llama-3.1-70B-Instruct'.

        token : str, optional
            The Hugging Face API token for authentication.

        split_token : str, optional
            The token used to split the text (default is '<split>').

        system_prompt : str, optional
            The system prompt to use for the model. If None, a default prompt is used.

        max_tokens : int, optional
            The maximum number of tokens to generate (default is 4096).

        stream : bool, optional
            Whether to stream the model's output (default is True).

        Raises
        ------
        ValueError
            If the model name is not provided.
        """
        if not model:
            raise ValueError("Model name must be provided.")

        self.client = InferenceClient(model, token=token)
        self.split_token = split_token
        self.max_tokens = max_tokens
        self.stream = stream

        self.system_prompt = system_prompt or self._default_system_prompt()

    def completion(self, text: str) -> str:
        """
        Call the language model to process the input text.

        This method sends the input text to the language model via the Hugging Face
        Inference API and returns the model's output.

        Parameters
        ----------
        text : str
            The input text to be processed by the model.

        Returns
        -------
        str
            The processed text returned by the model, potentially including split tokens.

        Raises
        ------
        RuntimeError
            If there's an error calling the Hugging Face Inference API.

        Notes
        -----
        - If streaming is enabled, the method will print the output in real-time
          and return the complete output as a string.
        - If streaming is disabled, the method will return the complete output
          after the model finishes processing.
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]

            if self.stream:
                return self._stream_completion(messages)

            else:
                return self._non_stream_completion(messages)

        except Exception as e:
            raise RuntimeError(f"Error calling Hugging Face Inference API: {str(e)}")

    def split(self, text: str) -> List[str]:
        """
        Split the input text into semantically coherent sections.

        This method sends the input text to the language model for processing,
        then splits the returned text based on the specified split token.

        Parameters
        ----------
        text : str
            The input text to be split.

        Returns
        -------
        List[str]
            A list of strings, each representing a semantically coherent section
            of the input text.

        Examples
        --------
        >>> splitter = SemanticTextSplitter(
        ...     model="meta-llama/Llama-2-70b-chat-hf",
        ...     token="your_hf_token_here"
        ... )
        >>> text = '''
        ... Machine learning is a subset of artificial intelligence
        ... that focuses on the development of algorithms and statistical models.
        ... It enables computer systems to improve their performance on a specific task
        ... through experience, without being explicitly programmed.
        ... On the other hand, deep learning is a subset of machine learning
        ... that uses artificial neural networks with multiple layers
        ... to progressively extract higher-level features from raw input.
        ... '''
        >>> result = splitter.split(text)
        >>> for idx, section in enumerate(result, 1):
        ...     print(f"Section {idx}:\n{section}\n")
        Section 1:
        Machine learning is a subset of artificial intelligence
        that focuses on the development of algorithms and statistical models.
        It enables computer systems to improve their performance on a specific task
        through experience, without being explicitly programmed.

        Section 2:
        On the other hand, deep learning is a subset of machine learning
        that uses artificial neural networks with multiple layers
        to progressively extract higher-level features from raw input.

        Notes
        -----
        - The quality of the splitting depends on the language model's understanding
          of the text and its ability to identify semantic boundaries.
        - The method uses the `completion` method internally to process the text,
          so any streaming behavior will occur during this step.
        - Empty sections (after stripping whitespace) are automatically removed
          from the final output.
        """
        processed_text = self.completion(text=text)

        return [
            section.strip()
            for section in re.split(f"{re.escape(self.split_token)}", processed_text)
            if section.strip()
        ]

    def _default_system_prompt(self) -> str:
        """
        Generate the default system prompt for the language model.

        This method creates a detailed instruction set for the language model,
        guiding it on how to split the input text into semantically coherent sections.

        Returns
        -------
        str
            The default system prompt as a string.

        Notes
        -----
        - This method is called internally if no custom system prompt is provided
          during initialization.
        - The prompt includes specific instructions on how to use the split token,
          handle different types of text, and maintain the integrity of the original content.
        """
        return f"""You are an assistant specialized in analyzing and dividing complex texts. Your task is to divide the provided text into semantically coherent sections, inserting the '{self.split_token}' tag between each distinct section. Follow these guidelines:

- Carefully analyze the semantic content of the text.
- Identify changes in theme, subject, or major concept.
- Insert the '{self.split_token}' tag at each point where you detect a significant change in semantic content.
- Ensure that each resulting section is self-contained and thematically coherent.
- Avoid dividing the text into sections that are too small or too numerous. Aim for divisions that capture complete ideas or concepts.
- Do not modify the original text apart from adding the '{self.split_token}' tags.
- Do not add explanations, comments, or additional metadata.
- If the text already contains natural divisions (such as paragraphs), use them as a guide, but don't hesitate to divide further if necessary for semantic coherence.
- Be consistent in your approach to division throughout the text.
- If the text is short or deals with a single coherent subject, it is acceptable not to divide it at all.
- Titles and subtitles should not be divided but only provide you with additional context.
- Correct format inconsistencies in the textual content if necessary, without modifying the text itself.
- You need to follow this instruction in all languages and always respond in the language of the provided text.

Your goal is to produce a version of the text divided in a way that facilitates subsequent labeling and analysis by language models. Focus solely on dividing the text and correcting the format according to these instructions, without adding any additional content."""

    def _stream_completion(self, messages: List[dict]) -> str:
        """
        Process the model's output in streaming mode.

        This method handles the streaming of the model's output, printing it
        in real-time and accumulating it into a single string.

        Parameters
        ----------
        messages : List[dict]
            A list of message dictionaries to be sent to the model.
            Each dictionary should have 'role' and 'content' keys.

        Returns
        -------
        str
            The complete output from the model as a single string.

        Notes
        -----
        - This method is called internally by the `completion` method when
          streaming is enabled.
        - It prints each chunk of the model's output as it's received, providing
          real-time feedback for long-running processes.
        """
        message = ""

        for chunk in self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
        ):
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                message += content

        return message

    def _non_stream_completion(self, messages: List[dict]) -> str:
        """
        Process the model's output in non-streaming mode.

        This method handles the model's output when streaming is disabled,
        returning the complete response at once.

        Parameters
        ----------
        messages : List[dict]
            A list of message dictionaries to be sent to the model.
            Each dictionary should have 'role' and 'content' keys.

        Returns
        -------
        str
            The complete output from the model as a single string.

        Notes
        -----
        - This method is called internally by the `completion` method when
          streaming is disabled.
        - It waits for the model to complete its entire response before returning,
          which may take longer for large inputs but provides the entire output at once.
        """
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            stream=False,
        )

        return response.choices[0].message.content


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