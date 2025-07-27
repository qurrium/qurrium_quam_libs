"""Qurrium Quam-Libs Crossroads Documentation Configuration"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# pylint: disable=invalid-name,redefined-builtin

from typing import Any

project = "Qurrium Quam-Libs Crossroads"
author = "Huai-Chung Chang (harui2019)"
copyright = "2025, Huai-Chun Chang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxext.opengraph",
    "myst_nb",  # required for JupyterBook-style notebooks
]

# -- Intersphinx mapping ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#module

intersphinx_mapping = {
    "qiskit:": (
        "https://quantum.cloud.ibm.com/docs/api/qiskit/",
        None,
    ),  # Qiskit documentation
    "qiskit-ibm-runtime": (
        "https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/",
        None,
    ),
    "qiskit-aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    # "tqdm": ("https://tqdm.github.io/", None),
    # Well, tqdm is not yet documented for intersphinx
    # See: https://github.com/tqdm/tqdm/issues/705
    "qurrium": ("https://docs.qurrium.org/", None),
}


# Execution settings (via myst-nb or jupyter-book)
nb_execution_mode = "force"
nb_execution_timeout = 1200
jupyter_execute_notebooks_only_build_toc_files = True  # pseudo-flag; handled by jupyter-book

templates_path = ["_templates"]
exclude_patterns = ["jupyter_execute"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = ""
html_favicon = ""
html_title = "Qurrium Quam-Libs Crossroads 🚏"
html_sourcelink_suffix = ""

html_extra_path = []  # You can set this if needed
html_static_path = ["_static"]
html_css_files = [
    "https://docs.qurrium.org/_static/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_theme_options: dict[str, Any] = {
    "light_css_variables": {
        "color-brand-primary": "#D55E00",
        "color-brand-content": "#F8921D",
        "color-brand-visited": "#348ABD",
    },
    "dark_css_variables": {
        "color-brand-primary": "#DA9057",
        "color-brand-content": "#FDB462",
        "color-brand-visited": "#81b1d2",
    },
}

html_last_updated_fmt = "%Y-%m-%d %H:%M:%S"  # Format for the last updated time
html_use_edit_page_button = False
html_use_repository_button = False
html_use_issues_button = False
html_use_multitoc_numbering = True
html_extra_footer = ""
html_home_page_in_navbar = True
html_announcement = ""


# -- Options for Pygments (syntax highlighting) -----------------------------

# pygments_style = "sphinx"
# pygments_dark_style = "lightbulb"
