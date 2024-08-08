ragoon
======

.. toctree::
    :maxdepth: 2
    :caption: 📚 Modules
    :glob:

    api/ragoon.chunks
    api/ragoon.datasets
    api/ragoon.embeddings
    api/ragoon.similarity_search
    api/ragoon.web_rag

This module provides the following submodules:

- `ragoon.chunks`: Contains functions for handling text chunks.
- `ragoon.datasets`: Contains functions for loading datasets concurrently.
- `ragoon.embeddings`: Provides methods for working with text embeddings.
- `ragoon.similarity_search`: Implements algorithms for similarity search.
- `ragoon.web_rag`: Offers functionality for web-based retrieval augmented generation.

Please refer to the individual submodules for detailed documentation and usage instructions.

Key Features
------------
- **Query Generation**: RAGoon generates search queries tailored to retrieve results that directly address the user's intent, enhancing the context for subsequent language model interactions.
- **Web Scraping and Data Retrieval**: RAGoon leverages web scraping capabilities to extract relevant content from various websites, providing language models with domain-specific knowledge.
- **Parallel Processing**: RAGoon utilizes parallel processing techniques to efficiently scrape and retrieve data from multiple URLs simultaneously.
- **Language Model Integration**: RAGoon integrates with language models, such as OpenAI's GPT-3 or LLama 3 on Groq Cloud, enabling users to leverage natural language processing capabilities for their applications.
- **Extensible Design**: RAGoon's modular architecture allows for the integration of new data sources, retrieval methods, and language models, ensuring future extensibility.