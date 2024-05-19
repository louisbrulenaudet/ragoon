# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dotenv import load_dotenv

from groq import Groq
from openai import OpenAI

from retrieval import Retriever
from scrape import WebScraper
from search import GoogleSearch


# Load environment variables from .env file
load_dotenv()

class RAGoon:
    def __init__(
        self,
        google_api_key:str,
        google_cx:str,
        completion_client,
        user_agent:str="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
    ) -> None:
        """
        Retrieval Augmented Goon (RAGoon) class.

        This class facilitates retrieval-based querying and completion using various APIs.

        Parameters
        ----------
        google_api_key : str
            The API key for Google services.
        
        google_cx : str
            The custom search engine ID for Google Custom Search.
        
        completion_client: str
            The API client for the completion service (e.g., OpenAI's GPT-3).
        
        user_agent : str, optional
            The user agent string to be used in web requests. Default is a Chrome user agent.

        Attributes
        ----------
        web_scraper : WebScraper
            An instance of the WebScraper class for web scraping.
        
        retriever : Retriever
            An instance of the Retriever class for data retrieval.
        
        google_search : GoogleSearch
            An instance of the GoogleSearch class for Google searches.

        """
        self.web_search = GoogleSearch(
            developer_key=google_api_key, 
            cx=google_cx
        )
        self.retriever = Retriever(
            client=completion_client
        )
        self.web_scraper = WebScraper(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        )


    def search(
        self,
        query:str,
        completion_model:str,
        system_prompt:str="""
        Given the user's input query, generate a concise and relevant Google search
        query that directly addresses the main intent of the user's question. The search query must
        be specifically tailored to retrieve results that can significantly enhance the context for a
        subsequent dialogue with an LLM. This approach will facilitate few-shot learning by providing
        rich, specific, and contextually relevant information. Please ensure that the response is
        well-formed and format it as a JSON object with a key named 'search_query'. This
        structured approach will help in assimilating the fetched results into an enhanced conversational
        model, contributing to a more nuanced and informed interaction.
        """,
        *args,
        **kargs
    ):
        """
        Search for information and perform completion.

        This method searches for information related to the given query
        and performs completion using the specified model. Additional
        parameters can be passed to the completion method.

        Parameters
        ----------
        query : str
            The search query.
        
        completion_model : str
            The name or identifier of the completion model to be used.
        
        *args
            Additional positional arguments to be passed to the completion method.
        
        **kwargs
            Additional keyword arguments to be passed to the completion method.

        Returns
        -------
        completion_data : dict
            A dictionary containing the generated completion data.

        """
        search_query = self.retriever.completion(
            model=completion_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            *args
        )["search_query"]

        search_results = self.web_search.search(
            query=search_query
        )

        list_of_urls = []

        for result in search_results:
            list_of_urls.append(result.get("link"))

        results = self.web_scraper.parallel_scrape(
            urls=list_of_urls,
            element_selectors= [
                "main"
            ]
        )

        return results


if __name__ == "__main__":
    instance = RAGoon(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cx=os.getenv("GOOGLE_CX"),
        completion_client=Groq(api_key=os.getenv("GROQ_API_KEY"))
    )

    instance.search(
        query="I want to do a left join in python polars",
        completion_model="Llama3-70b-8192",
        max_tokens=512,
        temperature=1,
    )