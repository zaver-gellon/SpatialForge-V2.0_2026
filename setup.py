"""Backward-compatible editable install for older pip/setuptools."""

from setuptools import find_packages, setup

setup(packages=find_packages(where="src"), package_dir={"": "src"})
