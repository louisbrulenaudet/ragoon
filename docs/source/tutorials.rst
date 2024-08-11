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

	
Similarity search
-----------------
The SimilaritySearch class is instantiated with specific parameters to configure the embedding model and search infrastructure. The chosen model, louisbrulenaudet/tsdae-lemone-mbert-base, is likely a multilingual BERT model fine-tuned with TSDAE (Transfomer-based Denoising Auto-Encoder) on a custom dataset. This model choice suggests a focus on multilingual capabilities and improved semantic representations.

The cuda device specification leverages GPU acceleration, crucial for efficient processing of large datasets. The embedding dimension of 768 is typical for BERT-based models, representing a balance between expressiveness and computational efficiency. The ip (inner product) metric is selected for similarity comparisons, which is computationally faster than cosine similarity when vectors are normalized. The i8 dtype indicates 8-bit integer quantization, a technique that significantly reduces memory usage and speeds up similarity search at the cost of a small accuracy rade-off.

.. code-block:: python

    from ragoon import (
        dataset_loader,
        SimilaritySearch,
        EmbeddingsVisualizer
    )
    instance = SimilaritySearch(
        model_name="louisbrulenaudet/tsdae-lemone-mbert-base",
        device="cuda",
        ndim=768,
        metric="ip",
        dtype="i8"
    )

The encode method transforms raw text into dense vector representations. This process involves tokenization, where text is split into subword units, followed by passing these tokens through the neural network layers of the SentenceTransformer model. The resulting embeddings capture semantic information in a high-dimensional space, where similar concepts are positioned closer together. The method likely uses batching to efficiently process large datasets and may employ techniques like length sorting to optimize padding and reduce computational waste.

.. code-block:: python

    dataset = dataset_loader(
        name="