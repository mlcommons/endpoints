# Sphinx Documentation

This directory contains the Sphinx documentation setup for generating and hosting API documentation with GitHub Pages.

## 📋 TODO: Documentation Setup

The following tasks need to be completed to enable full documentation:

### 1. Initialize Sphinx Structure

```bash
cd docs/sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser

# Run sphinx-quickstart
sphinx-quickstart
```

Configuration options:

- Separate source and build directories: Yes
- Project name: MLPerf Inference Endpoint
- Author name: NVIDIA
- Project release: 0.1.0
- Project language: en

### 2. Configure Sphinx

Edit `source/conf.py`:

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
```

### 3. Create Documentation Structure

Create these files in `source/`:

- `index.rst` - Main landing page
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start guide
- `cli.rst` - CLI reference
- `api.rst` - API reference (autodoc)
- `examples.rst` - Examples and tutorials
- `contributing.rst` - Development guide

### 4. Set Up GitHub Actions

Create `.github/workflows/docs.yml`:

```yaml
name: Build and Deploy Docs

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -e .
          pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser

      - name: Build documentation
        run: |
          cd docs/sphinx
          make html

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/sphinx/build/html
```

### 5. Configure GitHub Pages

1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

### 6. Add Sphinx Dependencies

Add to `requirements/dev.txt`:

```
sphinx>=7.0.0
sphinx-rtd-theme>=2.0.0
sphinx-autodoc-typehints>=1.25.0
myst-parser>=2.0.0
```

### 7. Build Locally

```bash
# Install dependencies
pip install -r requirements/dev.txt

# Build HTML documentation
cd docs/sphinx
make html

# View documentation
open build/html/index.html  # On macOS
# or
xdg-open build/html/index.html  # On Linux
# or
start build/html/index.html  # On Windows
```

### 8. Add Documentation Badge

Add to README.md:

```markdown
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://github.com/mlcommons/endpoints)
```

## 📚 Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [MyST Parser (Markdown)](https://myst-parser.readthedocs.io/)
- [Autodoc Tutorial](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

## 🎯 Current Status

- [ ] Sphinx structure initialized
- [ ] Configuration completed
- [ ] RST files created
- [ ] GitHub Actions workflow added
- [ ] GitHub Pages configured
- [ ] Local build tested
- [ ] Documentation badge added

## 📝 Notes

- Documentation will be automatically built and deployed on every push to `main`
- Preview builds are available for PRs
- Use Google/NumPy style docstrings in code for best autodoc results
- MyST Parser allows writing documentation in Markdown if preferred over RST
