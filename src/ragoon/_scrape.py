# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from multiprocessing import Pool
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

import httpx

from bs4 import BeautifulSoup


class WebScraper:
    """
    A class for scraping textual content from webpages based on specified element selectors.
    
    This class provides methods for retrieving textual content from webpages, either individually or in parallel.
    
    Attributes
    ----------
    user_agent : str
        The User-Agent header to simulate a desktop browser.
        
    Methods
    -------
    scrape_content(url, element_selectors=['main'])
        Scrape the textual content from a webpage based on the specified element selectors.
        
    parallel_scrape(urls, element_selectors=['main'])
        Scrape the textual content from multiple webpages in parallel using multiprocessing.
    """
    def __init__(
        self, 
        user_agent: str
    ):
        """
        Initializes the WebScraper object with the User-Agent header.

        Parameters
        ----------
        user_agent : str
            The User-Agent header to simulate a desktop browser.
        """
        self.user_agent = user_agent


    def scrape_content(
        self,
        url: str, 
        element_selectors: List[str] = ['main']
    ) -> str:
        """
        Scrape the textual content from a webpage based on the specified element selectors.

        Parameters
        ----------
        url : str
            The URL of the webpage to scrape.

        element_selectors : List[str], optional
            The CSS selector(s) of the HTML element(s) containing the desired text content.
            Default is ['main'].

        Returns
        -------
        str
            The textual content of the specified HTML element(s) on the webpage.
        """
        try:
            # Fetch webpage; simulate desktop browser
            with httpx.Client() as client:
                headers = {"User-Agent": self.user_agent}
                response = client.get(url, headers=headers)
                response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract text content for each selector
            content = []

            for selector in element_selectors:
                selected_elements = soup.select(selector)

                for element in selected_elements:
                    content.append(element.get_text(separator="\n", strip=True))

            # Join and return the text content
            return "\n".join(content)

        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")
            return ""
            

    def parallel_scrape(
        self, 
        urls: List[str], 
        element_selectors: List[str] = ['main']
    ) -> List[str]:
        """
        Scrape the textual content from multiple webpages in parallel using multiprocessing.

        Parameters
        ----------
        urls : List[str]
            A list of URLs of the webpages to scrape.

        element_selectors : List[str], optional
            The CSS selector(s) of the HTML element(s) containing the desired text content.
            Default is ['main'].

        Returns
        -------
        List[str]
            A list of textual content scraped from the webpages.
        """
        with Pool() as pool:
            results = pool.starmap(self.scrape_content, [(url, element_selectors) for url in urls])

        return results