# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import sys
import platform
import resource
import ctypes

from functools import wraps
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

from ragoon._logger import Logger

logger = Logger()


def get_memory_usage():
    """
    Get the memory usage of the current process.
    
    Returns
    -------
    memory_usage : int
        Memory usage in bytes.
    """
    if platform.system() == "Windows":
        process = ctypes.windll.kernel32.GetCurrentProcess()
        kernel32 = ctypes.windll.kernel32
        memory_info = ctypes.c_ulong()
        kernel32.GetProcessMemoryInfo(process, ctypes.byref(memory_info), ctypes.sizeof(memory_info))
        
        return memory_info.value
        
    else:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)


def memory(
    print_report=True
):
    """
    A decorator to monitor the memory usage of a function.
    
    Parameters
    ----------
    print_report : bool, optional
        If True, prints the memory usage report after the function completes. Default is True.
    
    Returns
    -------
    wrapper : function
        The decorated function with added memory monitoring functionality.
    """
    def decorator(
        func
    ):
        @wraps(func)
        def wrapper(
            *args,
            **kwargs
        ):
            """
            The wrapper function that monitors memory usage.
            
            Parameters
            ----------
            
            *args : tuple
                Positional arguments passed to the decorated function.
            
            **kwargs : dict
                Keyword arguments passed to the decorated function.
            
            Returns
            -------
            result : any
                The result of the decorated function.
            """
            global logger

            start_memory = get_memory_usage()

            result = func(*args, **kwargs)

            end_memory = get_memory_usage()
            memory_usage_mb = (end_memory - start_memory) / (1024.0 ** 2)

            if print_report:
                logger.info(f"Memory Usage Report for '{func.__name__}':")
                logger.info(f"  Memory Used: {memory_usage_mb:.2f} MB")

            return result

        return wrapper

    return decorator


def timer(
    print_time=True, 
    return_time=False
):
    """
    A decorator to measure the execution time of a function.
    
    Parameters
    ----------
    print_time : bool, optional
        If True, prints the execution time after the function completes. Default is True.
    
    return_time : bool, optional
        If True, returns a tuple containing the result of the function and the elapsed time. Default is False.
    
    Returns
    -------
    wrapper : function
        The decorated function with added timing functionality.
    """
    def decorator(
        func
    ):
        @wraps(func)
        def wrapper(
            *args, 
            **kwargs
        ):
            """
            The wrapper function that measures the execution time.
            
            Parameters
            ----------
            
            *args : tuple
                Positional arguments passed to the decorated function.
            
            **kwargs : dict
                Keyword arguments passed to the decorated function.
            
            Returns
            -------
            result : any
                The result of the decorated function.
            
            elapsed_time : float, optional
                If `return_time` is True, returns the elapsed time along with the result.
            """
            global logger
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if print_time:
                logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")

            if return_time:
                return result, elapsed_time

            return result

        return wrapper

    return decorator
