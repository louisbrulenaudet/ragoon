# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

import ragoon

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RAGoon'
copyright = '2024, Louis Brulé Naudet'
author = 'Louis Brulé Naudet'
release = '0.0.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
		'sphinx.ext.viewcode',
		'myst_parser'
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []

# Numpydoc configuration
numpydoc_show_class_members = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_default_options = {
    'member-order': 'bysource',
    'exclude-members': '__repr__, __str__, __weakref__',
    'members': True,             # Include all members (methods, attributes) of classes
    'undoc-members': True,       # Include members without docstrings
    'show-inheritance': True,    # Show class inheritance
    'special-members': '__init__',  # Include special methods like __init__
}

html_logo = '_static/images/logo_light.svg'  # Default logo
html_favicon = '_static/images/logo_light.svg'  # Default favicon

# Generate autosummary pages automatically
autosummary_generate = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

extensions.append('sphinx.ext.viewcode')

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/louisbrulenaudet/ragoon',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']

def setup(app):
    app.add_js_file('theme_switcher.js')