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

from ragoon._logger import Logger

logger = Logger()


def dataset_loader(
    name: str,
    streaming: bool=True
) -> datasets.Dataset:
    """
    Helper function to load a single dataset in parallel.

    Parameters
    ----------
    name : str
        Name of the dataset to be loaded.

    streaming : bool, optional
        Determines if datasets are streamed. Default is True.

    Returns
    -------
    dataset : datasets.Dataset
        Loaded dataset object.

    Raises
    ------
    Exception
        If an error occurs during dataset loading.
    """
    global logger

    try:
        return datasets.load_dataset(
            name,
            split="train",
            streaming=streaming
        )

    except Exception as exc:
        logger.error(f"Error loading dataset {name}: {exc}")

        return None


def load_datasets(
    req: list,
    streaming: bool=True
) -> list:
    """
    Downloads datasets specified in a list and creates a list of loaded datasets.

    Parameters
    ----------
    req : list
        A list containing the names of datasets to be downloaded.

    streaming : bool, optional
        Determines if datasets are streamed. Default is True.

    Returns
    -------
    datasets_list : list
        A list containing loaded datasets as per the requested names provided in 'req'.

    Raises
    ------
    Exception
        If an error occurs during dataset loading or processing.

    Examples
    --------
    >>> datasets = load_datasets(["dataset1", "dataset2"], streaming=False)
    """
    global logger
    
    datasets_list: str = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dataset = {executor.submit(dataset_loader, name, streaming): name for name in req}

        for future in tqdm(concurrent.futures.as_completed(future_to_dataset), total=len(req)):
            name = future_to_dataset[future]

            try:
                dataset = future.result()

                if dataset:
                    datasets_list.append(dataset)

            except Exception as exc:
                logger.error(f"Error processing dataset {name}: {exc}")

    return datasets_list