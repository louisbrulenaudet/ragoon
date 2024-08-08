# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from googleapiclient.discovery import build


class GoogleSearch:
    """
    A class to interact with the Google Custom Search API.

    Attributes
    ----------
    developer_key : str
        The developer key for accessing the API.

    cx : str
        The custom search engine ID.

    service : object
        The service object for interacting with the API.

    Methods
    -------
    __init__(developer_key, cx)
        Initializes the GoogleSearch object with the developer key and custom search engine ID (CX).

    search(query)
        Perform a Google search using the Custom Search API.
    """
    def __init__(
        self, 
        developer_key:str, 
        cx:str
    ) -> None:
        """
        Initializes the GoogleSearch object.

        Parameters
        ----------
        developer_key : str
            The developer key for accessing the API.

        cx : str
            The custom search engine ID.
        """
        self.service = build(
            "customsearch", 
            "v1", 
            developerKey=developer_key
        )

        self.cx = cx


    def search(
        self, 
        query:str
    ) -> list:
        """
        Perform a Google search using the Custom Search API.

        Parameters
        ----------
        query : str
            The search query.

        Returns
        -------
        list
            A list of search results as dictionaries.
        """
        try:
            # Execute the search query
            res = self.service.cse().list(
                q=query, 
                cx=self.cx
            ).execute()
            
            # Return the search results
            return res.get('items', [])
        
        except Exception as e:
            # Handle exceptions gracefully
            print(f"An error occurred: {e}")
            return []