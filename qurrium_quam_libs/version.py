"""Version (:mod:`qurrium_quam_libs.version`)"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, "VERSION.txt"), encoding="utf-8") as version_file:
    VERSION = version_file.read().strip()


__version__ = VERSION
"""Version of Qurrium Quam-Libs Crossroads."""

version_info = {"version": VERSION}
