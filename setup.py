from setuptools import setup

import os
import sys

install_requires = [
    "numpy>=1.14.0",
    "scipy>=1.1.0",
    "pygsp>=0.5.1",
    "scikit-learn>=0.20.0",
    "future",
    "tasklogger>=1.0",
    "Deprecated",
]

test_requires = [
    "pytest",
    "pytest-cov",
    "pandas",
    "coverage",
    "coveralls",
    "python-igraph",
    "parameterized",
    "anndata",
]

if sys.version_info[0] == 3:
    test_requires += ["anndata"]

doc_requires = ["sphinx", "sphinxcontrib-napoleon", "sphinxcontrib-bibtex"]

# Optional dependencies for performance acceleration
numba_requires = ["numba>=0.50.0"]

# Convenience extras
fast_requires = numba_requires  # For performance acceleration
all_requires = test_requires + doc_requires + numba_requires

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >=3.5 required.")
elif sys.version_info[:2] >= (3, 6):
    test_requires += ["black"]

version_py = os.path.join(os.path.dirname(__file__), "graphtools", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.rst").read()

setup(
    name="graphtools",
    version=version,
    description="graphtools",
    author="Scott Gigante, Daniel Burkhardt, and Jay Stanley, Yale University",
    author_email="scott.gigante@yale.edu",
    maintainer="João Felipe Rocha (Yale University) and Matthew Scicluna (Université de Montréal)",
    maintainer_email="joaofelipe.rocha@yale.edu, matthew.scicluna@umontreal.ca",
    packages=[
        "graphtools",
    ],
    license="GNU General Public License Version 2",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "doc": doc_requires,
        "numba": numba_requires,
        "fast": fast_requires,
        "all": all_requires,
    },
    long_description=readme,
    url="https://github.com/KrishnaswamyLab/graphtools",
    download_url="https://github.com/KrishnaswamyLab/graphtools/archive/v{}.tar.gz".format(
        version
    ),
    keywords=[
        "graphs",
        "big-data",
        "signal processing",
        "manifold-learning",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
