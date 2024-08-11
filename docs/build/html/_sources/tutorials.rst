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
        name="louisbrulenaudet/dac6-instruct",
        streaming=False,
        split="train"
    )

    dataset.save_to_disk("dataset.hf")
    embeddings = instance.encode(corpus=dataset["output"])

Binary quantization is an extreme form of dimensionality reduction, where each dimension of the embedding is represented by a single bit. This process involves setting a threshold (often the median value for each dimension across the dataset) and encoding values above this threshold as 1 and below as 0. While this dramatically reduces memory usage (compressing each embedding to just 96 bytes for a 768-dimensional vector), it also results in a more significant loss of information compared to other quantization methods. However, it enables extremely fast similarity computations using hardware-accelerated bitwise operations.

.. code-block:: python

    ubinary_embeddings = instance.quantize_embeddings(
        embeddings=embeddings,
        quantization_type="ubinary"
    )

Int8 quantization maps the continuous embedding values to a discrete set of 256 values represented by 8-bit integers. This process typically involves scaling the original values to fit within the int8 range (-128 to 127) and may use techniques like asymmetric quantization to preserve more information. While less extreme than binary quantization, int8 still offers substantial memory savings (reducing each dimension to 1 byte) while preserving more of the original information. This quantization enables efficient SIMD (Single Instruction, Multiple Data) operations on modern CPUs, significantly accelerating similarity computations.

.. code-block:: python

    int8_embeddings = instance.quantize_embeddings(
        embeddings=embeddings,
        quantization_type="int8"
    )

USEARCH is designed for high-performance approximate nearest neighbor search. The index creation process likely involves building a hierarchical structure, possibly a navigable small world (NSW) graph, which allows for efficient traversal during search operations. The use of int8 quantized embeddings enables USEARCH to leverage SIMD instructions for rapid distance calculations. The resulting index balances search speed and accuracy, allowing for fast retrieval with a controlled trade-off in precision.

.. code-block:: python

    instance.create_usearch_index(
        int8_embeddings=int8_embeddings,
        index_path="./usearch_int8.index",
        save=True
    )


FAISS (Facebook AI Similarity Search) is a library that provides efficient similarity search and clustering of dense vectors. For binary vectors, FAISS typically uses specialized index structures like the BinaryFlat index. This index performs exhaustive search using Hamming distance, which can be computed extremely efficiently on modern hardware using XOR and bit count operations. The binary nature of the index allows for compact storage and very fast search operations, albeit with reduced granularity in similarity scores compared to float-based indices.

.. code-block:: python

    instance.create_faiss_index(
        ubinary_embeddings=ubinary_embeddings,
        index_path="./faiss_ubinary.index",
        save=True
    )

The search process combines the strengths of both USEARCH and FAISS indices. It likely first uses the binary FAISS index for a rapid initial filtering step, leveraging the efficiency of Hamming distance calculations. The top candidates from this step (increased by the rescore_multiplier for better recall) are then refined using the more precise int8 USEARCH index. This two-stage approach balances speed and accuracy, allowing for quick pruning of unlikely candidates followed by more accurate rescoring.

The query is first encoded using the same model and quantization processes as the corpus. The rescore_multiplier of 4 means the initial retrieval fetches 40 candidates (4 * top_k), which are then reranked to produce the final top 10 results. This oversampling helps mitigate the potential loss of relevant results due to quantization approximations.

.. code-block:: python
    
    import polars as pl

    top_k_scores, top_k_indices = instance.search(
        query="Définir le rôle d'un intermédiaire concepteur conformément à l'article 1649 AE du Code général des Impôts.",
        top_k=10,
        rescore_multiplier=4
    )

    try:
        dataframe = pl.from_arrow(dataset.data.table).with_row_index()

    except:
        dataframe = pl.from_arrow(dataset.data.table).with_row_count(
            name="index"
        )

    scores_df = pl.DataFrame(
            {
                "index": top_k_indices,
                "score": top_k_scores
            }
    ).with_columns(
        pl.col("index").cast(pl.UInt32)
    )

    search_results = dataframe.filter(
        pl.col("index").is_in(top_k_indices)
    ).join(
        scores_df,
        how="inner",
        on="index"
    )

    search_results

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