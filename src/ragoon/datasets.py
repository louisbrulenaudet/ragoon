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

import datasets

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

from datasets import load_dataset
from tqdm import tqdm

from ragoon._logger import Logger

logger = Logger()


def dataset_loader(
    name: str, 
    streaming: Optional[bool] = True,
    split: Optional[Union[str, List[str]]] = None
) -> datasets.Dataset:
    """
    Helper function to load a single dataset in parallel.

    Parameters
    ----------
    name : str
        Name of the dataset to be loaded.

    streaming : bool, optional
        Determines if datasets are streamed. Default is True.

    split : Optional[Union[str, List[str]]], optional
        Which split of the data to load. If None, will return a dict with all splits (typically datasets.Split.TRAIN and datasets.Split.TEST). If given, will return a single Dataset. Splits can be combined and specified like in tensorflow-datasets.

    Returns
    -------
    dataset : datasets.Dataset
        Loaded dataset object.

    Raises
    ------
    Exception
        If an error occurs during dataset loading.
    """
    try:
        return load_dataset(
            name, 
            streaming=streaming,
            split=split
        )

    except Exception as exc:
        logger.error(f"Error loading dataset {name}: {exc}")

        return None


def load_datasets(
    req: list,
    streaming: Optional[bool] = False,
) -> list:
    """
    Downloads datasets specified in a list and creates a list of loaded datasets.

    Parameters
    ----------
    req : list
        A list containing the names of datasets to be downloaded.

    streaming : bool, optional
        Determines if datasets are streamed. Default is False.

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
    >>> req = [
    ...    "louisbrulenaudet/code-artisanat",
    ...    "louisbrulenaudet/code-action-sociale-familles",
    ... # ...
    ]

    >>> datasets_list = load_datasets(
    ...    req=req,
    ...    streaming=True
    )

    >>> dataset = datasets.concatenate_datasets(
    ...    datasets_list
    )
    """
    datasets_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dataset = {
            executor.submit(dataset_loader, name, streaming): name for name in req
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_dataset), total=len(req)
        ):
            name = future_to_dataset[future]

            try:
                dataset = future.result()

                if dataset:
                    datasets_list.append(dataset)

            except Exception as exc:
                logger.error(f"Error processing dataset {name}: {exc}")

    return datasets_list