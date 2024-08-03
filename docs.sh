#!/bin/bash

# Navigate to the docs directory
cd docs || { echo "Failed to change directory to 'docs'"; exit 1; }

# Generate reStructuredText files from the source code
sphinx-apidoc -o source/generated ../src/ragoon || { echo "sphinx-apidoc command failed"; exit 1; }

# Clean up any previous builds
make clean || { echo "make clean command failed"; exit 1; }

# Build the HTML documentation
make html || { echo "make html command failed"; exit 1; }

echo "Documentation successfully built."