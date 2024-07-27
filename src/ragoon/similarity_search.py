# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
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

import datasets
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from tqdm.notebook import tqdm
from usearch.index import Index

from ragoon._logger import Logger

logger = Logger()


class SimilaritySearch:
    """
    A class dedicated to encoding text data, quantizing embeddings, and managing indices for efficient similarity search.

    Attributes
    ----------
    model_name : str
        Name or identifier of the embedding model.

    device : str
        Computation device ('cpu' or 'cuda').

    ndim : int
        Dimension of the embeddings.

    metric : str
        Metric used for the index ('ip' for inner product, etc.).

    dtype : str
        Data type for the index ('i8' for int8, etc.).

    Methods
    -------
    encode(corpus, normalize_embeddings=True)
        Encodes a list of text data into embeddings.

    quantize_embeddings(embeddings, quantization_type)
        Quantizes the embeddings for efficient storage and search.

    create_faiss_index(ubinary_embeddings, index_path)
        Creates and saves a FAISS binary index.

    create_usearch_index(int8_embeddings, index_path)
        Creates and saves a USEARCH integer index.

    load_usearch_index_view(index_path)
        Loads a USEARCH index as a view for memory-efficient operations.

    load_faiss_index(index_path)
        Loads a FAISS binary index for searching.

    search(query, top_k=10, rescore_multiplier=4)
        Performs a search operation against the indexed embeddings.

    Examples
    --------
    >>> instance = SimilaritySearch(
        model_name="louisbrulenaudet/tsdae-lemone-mbert-base",
        device="cuda",
        ndim=768,
        metric="ip",
        dtype="i8"
    )
    >>> embeddings = instance.encode(corpus=dataset["output"])
    >>> ubinary_embeddings = instance.quantize_embeddings(
        embeddings=embeddings,
        quantization_type="ubinary"
    )
    >>> int8_embeddings = instance.quantize_embeddings(
        embeddings=embeddings,
        quantization_type="int8"
    )
    >>> instance.create_usearch_index(
        int8_embeddings=int8_embeddings,
        index_path="./usearch_int8.index"
    )
    >>> instance.create_faiss_index(
        ubinary_embeddings=ubinary_embeddings,
        index_path="./faiss_ubinary.index"
    )
    >>> top_k_scores, top_k_indices = instance.search(
        query="Sont considérées comme ayant leur domicile fiscal en France au sens de l'article 4 A",
        top_k=10,
        rescore_multiplier=4
    )
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        ndim: int = 1024,
        metric: str = "ip",
        dtype: str = "i8"
    ):
        """
        Initializes the EmbeddingIndexer with the specified model, device, and index configurations.

        Parameters
        ----------
        model_name : str
            The name or identifier of the SentenceTransformer model to use for embedding.

        device : str, optional
            The computation device to use ('cpu' or 'cuda'). Default is 'cuda'.

        ndim : int, optional
            The dimensionality of the embeddings. Default is 1024.

        metric : str, optional
            The metric used for the index ('ip' for inner product). Default is 'ip'.

        dtype : str, optional
            The data type for the USEARCH index ('i8' for 8-bit integer). Default is 'i8'.
        """
        self.model_name: str = model_name
        self.device: str = device
        self.ndim: int = ndim
        self.metric: str = metric
        self.dtype: str = dtype
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        self.binary_index = None
        self.int8_index = None


    def encode(
        self,
        corpus: list,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encodes the given corpus into full-precision embeddings.

        Parameters
        ----------
        corpus : list
            A list of sentences to be encoded.

        normalize_embeddings : bool, optional
            Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used. Default is True.

        Returns
        -------
        np.ndarray
            The full-precision embeddings of the corpus.

        Notes
        -----
        This method normalizes the embeddings and shows the progress bar during the encoding process.
        """
        global logger

        try:
            embeddings = self.model.encode(
                corpus,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=True
            )
            return embeddings

        except Exception as e:
            logger.error(f"An error occurred during encoding: {e}")


    def quantize_embeddings(
        self,
        embeddings: np.ndarray,
        quantization_type: str
    ) -> Union[np.ndarray, bytearray]:
        """
        Quantizes the given embeddings based on the specified quantization type ('ubinary' or 'int8').

        Parameters
        ----------
        embeddings : np.ndarray
            The full-precision embeddings to be quantized.
        quantization_type : str
            The type of quantization ('ubinary' for unsigned binary, 'int8' for 8-bit integers).

        Returns
        -------
        Union[np.ndarray, bytearray]
            The quantized embeddings.

        Raises
        ------
        ValueError
            If an unsupported quantization type is provided.
        """
        global logger

        try:
            if quantization_type == "ubinary":
                return self._quantize_to_ubinary(
                    embeddings=embeddings
                )

            elif quantization_type == "int8":
                return self._quantize_to_int8(
                    embeddings=embeddings
                )

            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")

        except Exception as e:
            logger.error(f"An error occurred during quantization: {e}")


    def create_faiss_index(
        self,
        ubinary_embeddings: bytearray,
        index_path: str = None,
        save: bool = False
    ) -> None:
        """
        Creates and saves a FAISS binary index from ubinary embeddings.

        Parameters
        ----------
        ubinary_embeddings : bytearray
            The ubinary-quantized embeddings.

        index_path : str, optional
            The file path to save the FAISS binary index. Default is None.

        save : bool, optional
            Indicator for saving the index. Default is False.

        Notes
        -----
        The dimensionality of the index is specified during the class initialization (default is 1024).
        """
        global logger

        try:
            self.binary_index = faiss.IndexBinaryFlat(
                self.ndim
            )
            self.binary_index.add(
                ubinary_embeddings
            )

            if save and index_path:
                self._save_faiss_index_binary(
                    index_path=index_path
                )

        except Exception as e:
            logger.error(f"An error occurred during index creation: {e}")


    def create_usearch_index(
        self,
        int8_embeddings: np.ndarray,
        index_path: str = None,
        save: bool = False
    ) -> None:
        """
        Creates and saves a USEARCH integer index from int8 embeddings.

        Parameters
        ----------
        int8_embeddings : np.ndarray
            The int8-quantized embeddings.

        index_path : str, optional
            The file path to save the USEARCH integer index. Default is None.

        save : bool, optional
            Indicator for saving the index. Default is False.

        Returns
        -------
        None

        Notes
        -----
        The dimensionality and metric of the index are specified during class initialization.
        """
        global logger

        try:
            self.int8_index = Index(
                ndim=self.ndim,
                metric=self.metric,
                dtype=self.dtype
            )

            self.int8_index.add(
                np.arange(
                    len(int8_embeddings)
                ),
                int8_embeddings
            )

            if save == True and index_path:
                self._save_int8_index(
                    index_path=index_path
                )

            return self.int8_index

        except Exception as e:
            logger.error(f"An error occurred during USEARCH index creation: {e}")


    def load_usearch_index_view(
        self,
        index_path: str
    ) -> any:
        """
        Loads a USEARCH index as a view for memory-efficient operations.

        Parameters
        ----------
        index_path : str
            The file path to the USEARCH index to be loaded as a view.

        Returns
        -------
        object
            A view of the USEARCH index for memory-efficient similarity search operations.

        Notes
        -----
        Implementing this would depend on the specific USEARCH index handling library being used.
        """
        global logger

        try:
            self.int8_index = Index.restore(
                index_path,
                view=True
            )

            return self.int8_index

        except Exception as e:
            logger.error(f"An error occurred while loading USEARCH index: {e}")


    def load_faiss_index(
        self,
        index_path: str
    ) -> None:
        """
        Loads a FAISS binary index from a specified file path.

        This method loads a binary index created by FAISS into the class
        attribute `binary_index`, ready for performing similarity searches.

        Parameters
        ----------
        index_path : str
            The file path to the saved FAISS binary index.

        Returns
        -------
        None

        Notes
        -----
        The loaded index is stored in the `binary_index` attribute of the class.
        Ensure that the index at `index_path` is compatible with the configurations
        (e.g., dimensions) used for this class instance.
        """
        global logger

        try:
            self.binary_index = faiss.read_index_binary(
                index_path
            )

        except Exception as e:
            logger.error(f"An error occurred while loading the FAISS index: {e}")


    def search(
        self,
        query: str,
        top_k: int = 10,
        rescore_multiplier: int = 4
    ) -> Tuple[List[float], List[int]]:
        """
        Performs a search operation against the indexed embeddings.

        Parameters
        ----------
        query : str
            The query sentence/string to be searched.

        top_k : int, optional
            The number of top results to return.

        rescore_multiplier : int, optional
            The multiplier used to increase the initial retrieval size for re-scoring.
            Higher values can increase precision at the cost of performance.

        Returns
        -------
        Tuple[List[float], List[int]]
            A tuple containing the scores and the indices of the top k results.

        Notes
        -----
        This method assumes that `binary_index` and `int8_index` are already loaded or created.
        """
        global logger

        try:
            if self.binary_index is None or self.int8_index is None:
                raise ValueError("Indices must be loaded or created before searching.")

            query_embedding = self.encode(
                corpus=query,
                normalize_embeddings=False
            )

            query_embedding_ubinary = self.quantize_embeddings(
                embeddings=query_embedding.reshape(1, -1),
                quantization_type="ubinary"
            )

            _scores, binary_ids = self.binary_index.search(
                query_embedding_ubinary,
                top_k * rescore_multiplier
            )

            binary_ids = binary_ids[0]

            int8_embeddings = self.int8_index[binary_ids].astype(int)

            scores = query_embedding @ int8_embeddings.T

            indices = (-scores).argsort()[:top_k]

            top_k_indices = binary_ids[indices]
            top_k_scores = scores[indices]

            return top_k_scores.tolist(), top_k_indices.tolist()

        except Exception as e:
            logger.error(f"An error occurred while searching semantic similar sentences: {e}")


    def _quantize_to_ubinary(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Placeholder private method for ubinary quantization.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to quantize.

        Returns
        -------
        np.ndarray
            The quantized embeddings.
        """
        global logger

        try:
            ubinary_embeddings = quantize_embeddings(
                embeddings,
                "ubinary"
            )
            return ubinary_embeddings

        except Exception as e:
            logger.error(f"An error occurred during ubinary quantization: {e}")


    def _quantize_to_int8(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Placeholder private method for int8 quantization.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to quantize.

        Returns
        -------
        np.ndarray
            The quantized embeddings.
        """
        global logger

        try:
            int8_embeddings = quantize_embeddings(
                embeddings,
                "int8"
            )

            return int8_embeddings

        except Exception as e:
            logger.error(f"An error occurred during int8 quantization: {e}")


    def _save_faiss_index_binary(
        self,
        index_path: str
    ) -> None:
        """
        Saves the FAISS binary index to disk.

        This private method is called internally to save the constructed FAISS binary index to the specified file path.

        Parameters
        ----------
        index_path : str
            The path to the file where the binary index should be saved. This value is checked in the public method
            `create_faiss_index`.

        Returns
        -------
        None

        Notes
        -----
            This method should not be called directly. It is intended to be used internally by the `create_faiss_index` method.
        """
        global logger

        try:
            faiss.write_index_binary(
                self.binary_index,
                index_path
            )

            return None

        except Exception as e:
            logger.error(f"An error occurred during FAISS binary index saving: {e}")


    def _save_int8_index(
        self,
        index_path: str
    ) -> None:
        """
        Saves the int8_index to disk.

        This private method is called internally to save the constructed int8_index to the specified file path.

        Parameters
        ----------
        index_path : str
            The path to the file where the int8_index should be saved. This value is checked in the public method
            `_save_int8_index`.

        Returns
        -------
        None

        Notes
        -----
            This method should not be called directly. It is intended to be used internally by the `_save_int8_index` method.
        """
        global logger

        try:
            self.int8_index.save(
                index_path
            )

            return None

        except Exception as e:
            logger.error(f"An error occurred during int8_index saving: {e}")