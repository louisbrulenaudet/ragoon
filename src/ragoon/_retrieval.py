# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

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


class Retriever:
    """
    A class for retrieving completions from an API client.

    This class provides methods for generating text completions using an API client,
    either synchronously or asynchronously.

    Attributes
    ----------
    client : object
        An instance of the API client, which can be from OpenAI, Groq, or any other similar service.

    Methods
    -------
    completion(model, messages, max_tokens, temperature=1, stream=True, top_p=1, frequency_penalty=0,
               presence_penalty=0, response_format={'type': 'text'})
        Generate a completion using the API.

    async_completion(model, messages, temperature=1.0, max_tokens=150, top_p=1, frequency_penalty=0,
                     presence_penalty=0, response_format='text')
        Asynchronously completes a chat conversation using the API.
    """
    def __init__(
        self, 
        client
    ):
        """
        Initialize the Retriever with an API client.

        Parameters
        ----------
        client : object
            An instance of the API client, which can be from OpenAI, Groq, or any other similar service.
        """
        self.client = client


    def completion(
        self, 
        model:str, 
        messages:list, 
        max_tokens:int=512, 
        temperature:float=1, 
        stream:bool=False, 
        top_p:float=1, 
        frequency_penalty:float=0, 
        presence_penalty:float=0, 
        response_format:dict={"type": "json_object"},
        *args
    ) -> dict:
        """
        Generate a completion using the API.

        Parameters
        ----------
        model : str
            The identifier of the model to be used for completion.

        messages : list
            List of strings representing the conversation history or context for the completion.

        max_tokens : int, optional
            The maximum number of tokens the completion should generate. Default is 512.

        temperature : float, optional
            Controls the randomness of the generated text. Higher values make the text more random. Default is 1.

        stream : bool, optional
            Determines whether the completion is streamed or not. Default is False.

        top_p : float, optional
            The nucleus sampling parameter. Default is 1.

        frequency_penalty : float, optional
            The frequency penalty parameter. Default is 0.

        presence_penalty : float, optional
            The presence penalty parameter. Default is 0.
            
        response_format : dict, optional
            The format in which the response is expected. Default is {'type': 'json_object'}.

        *args
            Additional positional arguments to be passed to the completion method.

        Returns
        -------
        completion_data : dict
            A dictionary containing the generated completion data.

        Notes
        -----
        This function interacts with the API to generate text completions based on the provided context.
        It creates a completion request with specified parameters such as model, messages, temperature, and max_tokens.
        The completion data is extracted and processed from the API response.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=stream,
                response_format=response_format
            )

            if stream:
                return self._process_streaming_response(response)
            else:
                return self._process_non_streaming_response(response)

        except Exception as e:
            print(f"An error occurred: {e}")
            return {}


    def _process_streaming_response(
        self, 
        response
    ) -> dict:
        """
        Process the streaming response from the API.

        Parameters
        ----------
        response : object
            The response object returned by the API.

        Returns
        -------
        dict
            A dictionary containing the cleaned streaming data.
        """
        cleaned_data = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                cleaned_data += content

        return json.loads(cleaned_data)


    def _process_non_streaming_response(
        self, 
        response
    ) -> dict:
        """
        Process the non-streaming response from the API.

        Parameters
        ----------
        response : object
            The response object returned by the API.

        Returns
        -------
        dict
            A dictionary containing the cleaned non-streaming data.
        """
        cleaned_data = response.choices[0].message.content
        return json.loads(cleaned_data)


    async def async_completion(
        self, 
        model:str,
        messages:list, 
        temperature:float=1.0, 
        max_tokens:int=512, 
        top_p:float=1, 
        frequency_penalty:float=0, 
        presence_penalty:float=0, 
        response_format:str="json_object"
    ) -> dict:
        """
        Asynchronously completes a chat conversation using the API.

        Parameters
        ----------
        model : str
            The name of the language model to use.

        messages : List[Dict[str, str]]
            A list of messages in the chat conversation. Each message is a dictionary with "role" (either "user" or "assistant")
            and "content" (the text of the message).

        temperature : float, optional
            Controls the randomness of the generated output. Higher values make the output more random. Default is 1.0.

        max_tokens : int, optional
            Limits the length of the generated output to the specified number of tokens. Default is 512.

        top_p : float, optional
            The nucleus sampling parameter. Default is 1.

        frequency_penalty : float, optional
            The frequency penalty parameter. Default is 0.

        presence_penalty : float, optional
            The presence penalty parameter. Default is 0.
            
        response_format : str, optional
            The format in which the response is expected. Default is "json_object".

        Returns
        -------
        completion_data : Dict
            A dictionary containing the completed chat conversation. The cleaned data is extracted from the API response.

        Notes
        -----
        If an error occurs during the API call, an empty dictionary is returned.
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                response_format=response_format
            )

            cleaned_data = response.choices[0].message.content
            
            return json.loads(cleaned_data)

        except Exception as e:
            print(f"An error occurred: {e}")
            return {}
