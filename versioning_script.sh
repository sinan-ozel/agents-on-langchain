#!/bin/bash

# Get the last Git tag (stable version)
echo $LAST_TAG

cd /build

# Extract the version from the Python package
VERSION=$(python -c "from agents_on_langchain import __version__; print(__version__)")

echo $VERSION

# Check if the version is different from the last tag
if [[ $VERSION != $LAST_TAG ]]; then
  # Create a new Git tag for stable releases
  git tag "$VERSION"
  git push origin "$VERSION"
else
  # For nightly/dev releases, append the commit hash to the version
  VERSION="$VERSION-dev$(git rev-parse --short HEAD)"
fi

# Build the package using setuptools
python -m build

# Upload the package to PyPI
if [[ $VERSION == *"-dev"* ]]; then
  # For nightly/dev releases, upload to the PyPI repository
  twine upload --repository pypi dist/* -u __token__ -p "$PYPI_TOKEN"
else
  # For stable releases, upload to PyPI
  twine upload dist/*  -u __token__ -p "$PYPI_TOKEN"
fi
