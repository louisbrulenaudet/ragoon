![Plot](https://github.com/louisbrulenaudet/ragoon/blob/main/thumbnail.png?raw=true)

# RAGoon : Improve Large Language Models retrieval using dynamic web-search ⚡
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)
<a target="_blank" href="https://colab.research.google.com/github/louisbrulenaudet/tax-retrieval-benchmark/blob/main/RAGoon%20%3A%20Improve%20Large%20Language%20Models%20retrieval%20using%20dynamic%20web-search.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

RAGoon is a Python library that aims to improve the performance of language models by providing contextually relevant information through retrieval-based querying, web scraping, and data augmentation techniques. It offers an integration of various APIs, enabling users to retrieve information from the web, enrich it with domain-specific knowledge, and feed it to language models for more informed responses.

RAGoon's core functionality revolves around the concept of few-shot learning, where language models are provided with a small set of high-quality examples to enhance their understanding and generate more accurate outputs. By curating and retrieving relevant data from the web, RAGoon equips language models with the necessary context and knowledge to tackle complex queries and generate insightful responses.

## Usage Example
Here's an example of how to use RAGoon:

```python
from groq import Groq
# from openai import OpenAI
from ragoon import RAGoon

# Initialize RAGoon instance
ragoon = RAGoon(
    google_api_key="your_google_api_key",
    google_cx="your_google_cx",
    completion_client=Groq(api_key="your_groq_api_key")
)

# Search and get results
query = "I want to do a left join in python polars"
results = ragoon.search(
    query=query,
    completion_model="Llama3-70b-8192",
    max_tokens=512,
    temperature=1,
)

# Print results
print(results)
```

## Key Features
- **Query Generation**: RAGoon generates search queries tailored to retrieve results that directly address the user's intent, enhancing the context for subsequent language model interactions.
- **Web Scraping and Data Retrieval**: RAGoon leverages web scraping capabilities to extract relevant content from various websites, providing language models with domain-specific knowledge.
- **Parallel Processing**: RAGoon utilizes parallel processing techniques to efficiently scrape and retrieve data from multiple URLs simultaneously.
- **Language Model Integration**: RAGoon integrates with language models, such as OpenAI's GPT-3 or LLama 3 on Groq Cloud, enabling users to leverage natural language processing capabilities for their applications.
- **Extensible Design**: RAGoon's modular architecture allows for the integration of new data sources, retrieval methods, and language models, ensuring future extensibility.

## Dependencies
- `dotenv`: A Python library that loads environment variables from a `.env` file.
- `groq`: A Python client library for the Groq API, which provides access to language models.
- `openai`: A Python library to interact with the OpenAI API, including access to GPT-3 and other language models.
- `requests`: A popular Python library for making HTTP requests and interacting with web services.
- `beautifulsoup4`: A Python library for web scraping, providing tools for parsing and navigating HTML and XML documents.
- `httpx`: A modern, Python 3 library for making HTTP requests.
- `googleapiclient`: A Python client library for accessing Google APIs.

## Citing this project
If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2024,
	author = {Louis Brulé Naudet},
	title = {RAGoon : Improve Large Language Models retrieval using dynamic web-search},
	howpublished = {\url{https://github.com/louisbrulenaudet/ragoon}},
	year = {2024}
}
```
## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).
