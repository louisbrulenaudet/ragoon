# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/ragoon'))

project = 'RAGoon'
copyright = '2024, Louis Brulé Naudet'
author = 'Louis Brulé Naudet'
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
    "external_links": [
        {"name": "GitHub", "url": "https://github.com/louisbrulenaudet"},
        {"name": "Twitter", "url": "https://x.com/louisbrulenaudet"}
    ],
    # "logo": {
    #     "text": "RAGoon",
    #     "image_light": "_static/your_logo.png",
    #     "image_dark": "_static/your_logo.png"
    # }
}