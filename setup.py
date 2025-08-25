"""Setup Script for Qurrium Quam-Libs Crossroads"""

import os
from setuptools import setup


with open(os.path.join("qurrium_quam_libs", "VERSION.txt"), encoding="utf-8") as version_file:
    __version__ = version_file.read().strip()

print(f"| Version: {__version__}")


setup(version=__version__)
