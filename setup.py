from setuptools import setup, find_packages
import pathlib

## Template from https://github.com/pypa/sampleproject/blob/main/setup.py

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="lapgm",
    version="0.1.3",
    description="A spatially regularized Gaussian mixture model for "
                "MR bias field correction and intensity normalization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucianoAvinas/lapgm",
    author="Luciano Vinas",
    author_email="lucianovinas@g.ucla.edu",
    license='MIT',
    packages=find_packages(exclude=['tests','examples']),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20",
        "scipy",
        "scikit-learn",
        "matplotlib"
    ],
    extras_require={"gpu": ["cupy-cuda11x"]
    },
)