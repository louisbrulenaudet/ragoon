# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RAGoon'
copyright = '2024, Louis Brulé Naudet'
author = 'Louis Brulé Naudet'
release = '0.0.4'

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'RAGoon'
author = 'Your Name'
release = '0.0.4'

extensions = [
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    # "logo": {
    #     "text": "RAGoon",
    #     "image_light": "_static/your_logo.png",
    #     "image_dark": "_static/your_logo.png"
    # },
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/yourprofile", "icon": "fab fa-github"},
        {"name": "Twitter", "url": "https://twitter.com/yourprofile", "icon": "fab fa-twitter"},
        {"name": "LinkedIn", "url": "https://linkedin.com/in/yourprofile", "icon": "fab fa-linkedin"},
    ]
}
