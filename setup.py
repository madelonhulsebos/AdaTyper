"""Setup for AdaTyper."""

from setuptools import setup, find_packages

setup(
    name="adatyper",
    version="0.0.2",
    description="Interactive system for detecting semantic column types in tables.",
    author="Madelon Hulsebos",
    author_email="madelon@sigmacomputing.com",
    packages=find_packages(include=["typetabert", "adatyper", "table_bert"]),
)
