
from setuptools import find_namespace_packages, setup


# Installation operation
setup(
    name="sugar_rl",
    packages=find_namespace_packages(include=["sugar_rl*"]),
    version="0.1.0",
    install_requires=[],
    python_requires=">=3.11",
    zip_safe=False,
)