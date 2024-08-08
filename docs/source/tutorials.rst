🖼️ Tutorials
============

This section provides an overview of different code blocks that can be executed with RAGoon to enhance your NLP and language model projects.

Embeddings production
---------------------
This class handles loading a dataset from Hugging Face, processing it to add embeddings using specified models, and provides methods to save and upload the processed dataset.

.. code-block:: python

    from ragoon import EmbeddingsDataLoader
    from datasets import load_dataset

    # Initialize the dataset loader with multiple models
    loader = EmbeddingsDataLoader(
        token="hf_token",
        dataset=load_dataset("louisbrulenaudet/dac6-instruct", split="train"),  # If dataset is already loaded.
        # dataset_name="louisbrulenaudet/dac6-instruct",  # If you want to load the dataset from the class.
        model_configs=[
            {"model": "bert-base-uncased", "query_prefix": "Query:"},
            {"model": "distilbert-base-uncased", "query_prefix": "Query:"}
            # Add more model configurations as needed
        ]
    )

    # Uncomment this line if passing dataset_name instead of dataset.
    # loader.load_dataset()

    # Process the splits with all models loaded
    loader.process(
        column="output",
        preload_models=True
    )

    # To access the processed dataset
    processed_dataset = loader.get_dataset()
    print(processed_dataset[0])

You can also embed a single text using multiple models:

.. code-block:: python

    from ragoon import EmbeddingsDataLoader

    # Initialize the dataset loader with multiple models
    loader = EmbeddingsDataLoader(
        token="hf_token",
        model_configs=[
            {"model": "bert-base-uncased"},
            {"model": "distilbert-base-uncased"}
        ]
    )

    # Load models
    loader.load_models()

    # Embed a single text with all loaded models
    text = "This is a single text for embedding."
    embedding_result = loader.batch_encode(text)

    # Output the embeddings
    print(embedding_result)

Embeddings visualization
------------------------
This class provides functionality to load embeddings from a FAISS index, reduce their dimensionality using PCA and/or t-SNE, and visualize them in an interactive 3D plot.

.. code-block:: python

    from ragoon import EmbeddingsVisualizer

    visualizer = EmbeddingsVisualizer(
        index_path="path/to/index", 
        dataset_path="path/to/dataset"
    )

    visualizer.visualize(
        method="pca",
        save_html=True,
        html_file_name="embedding_visualization.html"
    )

Dynamic web search
------------------
RAGoon is a Python library that aims to improve the performance of language models by providing contextually relevant information through retrieval-based querying, web scraping, and data augmentation techniques. It integrates various APIs, enabling users to retrieve information from the web, enrich it with domain-specific knowledge, and feed it to language models for more informed responses.

RAGoon's core functionality revolves around the concept of few-shot learning, where language models are provided with a small set of high-quality examples to enhance their understanding and generate more accurate outputs. By curating and retrieving relevant data from the web, RAGoon equips language models with the necessary context and knowledge to tackle complex queries and generate insightful responses.

.. code-block:: python

    from groq import Groq
    # from openai import OpenAI
    from ragoon import WebRAG

    # Initialize RAGoon instance
    ragoon = WebRAG(
        google_api_key="your_google_api_key",
        google_cx="your_google_cx",
        completion_client=Groq(api_key="your_groq_api_key")
    )

    # Search and get results
    query = "I want to do a left join in Python Polars"
    results = ragoon.search(
        query=query,
        completion_model="Llama3-70b-8192",
        max_tokens=512,
        temperature=1,
    )

    # Print results
    print(results)