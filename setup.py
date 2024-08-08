# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import (
	setup, 
	find_packages
)

setup(
    name="ragoon",
    version="0.0.4",
    description="RAGoon: High level library for batched embeddings generation, blazingly-fast web-based RAG and quantitized indexes processing ⚡",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Louis Brulé Naudet",
    author_email="louisbrulenaudet@icloud.com",
    url="https://github.com/louisbrulenaudet/ragoon",
    project_urls={
        "Homepage": "https://github.com/louisbrulenaudet/ragoon",
        "Repository": "https://github.com/louisbrulenaudet/ragoon",
    },
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "language-models",
        "retrieval",
        "web-scraping",
        "few-shot-learning",
        "nlp",
        "machine-learning",
        "retrieval-augmented-generation",
        "RAG",
        "groq",
        "generative-ai",
        "llama",
        "Mistral",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "beautifulsoup4==4.12.3",
        "datasets==2.20.0",
        "faiss_cpu==1.8.0",
        "google_api_python_client==2.126.0",
        "groq==0.9.0",
        "httpx==0.27.0",
        "huggingface_hub==0.24.2",
        "myst-parser==3.0.1",
        "numpy<2",
        "numpydoc==1.7.0",
        "openai==1.37.1",
        "overload==1.1",
        "plotly==5.23.0",
        "pydata-sphinx-theme==0.15.4",
        "pytest==8.3.2",
        "scikit_learn==1.5.1",
        "sentence_transformers==3.0.1",
        "sphinx==7.4.7",
        "sphinx_book_theme==1.1.3",
        "torch==2.2.1",
        "transformers",
        "tqdm==4.66.4",
        "umap==0.1.1",
        "umap_learn==0.5.6",
        "usearch==2.12.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-book-theme>=1.0.1",
            "sphinxcontrib-katex",
            "sphinx-autodoc-typehints",
            "ipython>=8.8.0",
            "myst-nb>=1.0.0",
            "myst-parser",
            "matplotlib>=3.5.0",
            "sphinx_book_theme==1.1.3",
            "sphinx-gallery>=0.14.0",
            "sphinx-collections>=0.0.1",
            "tensorflow>=2.4.0",
            "tensorflow-datasets>=4.2.0",
            "flax",
            "sphinx_contributors",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # Add any command-line tools here
        ]
    },
)
