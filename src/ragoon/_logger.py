# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import time

from threading import Lock
from tqdm import tqdm
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


class Logger:
    """
    SimpleLogger is a utility class that provides a simple and beautiful logging interface.

    Attributes
    ----------
    logger : logging.Logger
        The logger instance.

    Methods
    -------
    info(message: str):
        Log an informational message.
    
    warning(message: str):
        Log a warning message.
    
    error(message: str):
        Log an error message.
    """
    def __init__(
        self
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


    def info(
        self,
        message: str
    ):
        """
        Log an informational message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.logger.info(message)


    def warning(
        self, 
        message: str
    ):
        """
        Log a warning message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.logger.warning(message)


    def error(
        self, 
        message: str
    ):
        """
        Log an error message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.logger.error(message)


class TqdmToLogger:
    """
    TqdmToLogger is a utility class that redirects tqdm output to a logger.

    Parameters
    ----------
    total : int, optional
        Total number of iterations for the progress bar, by default 100

    Attributes
    ----------
    pbar : tqdm.tqdm
        The progress bar instance.
    """
    def __init__(
        self, 
        total: int = 100
    ):
        self.pbar = tqdm(
            total=total
        )


    def write(
        self, 
        buf: str
    ):
        """
        Write method to handle logging.

        Parameters
        ----------
        buf : str
            The string buffer to write.
        """
        if buf.startswith("Construct"):
            self.pbar.update(1)

        self.pbar.set_description(buf.strip())

    def flush(self):
        """Flush method (does nothing but needed for interface compatibility)."""
        pass