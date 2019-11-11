import os
import sys
from setuptools import setup

install_requires = [
    "numpy>=1.14.0",
    "scipy>=1.1.0",
    "pygsp>=0.5.1",
    "scikit-learn>=0.20.0",
    "future",
    "tasklogger>=1.0",
]

test_requires = [
    "nose",
    "nose2",
    "pandas",
    "coverage",
    "coveralls",
    "python-igraph",
    "parameterized",
]

if sys.version_info[0] == 3:
    test_requires += ["anndata"]

doc_requires = ["sphinx", "sphinxcontrib-napoleon", "sphinxcontrib-bibtex"]

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
    packages=["graphtools",],
    license="GNU General Public License Version 2",
    install_requires=install_requires,
    extras_require={"test": test_requires, "doc": doc_requires},
    test_suite="nose2.collector.collector",
    long_description=readme,
    url="https://github.com/KrishnaswamyLab/graphtools",
    download_url="https://github.com/KrishnaswamyLab/graphtools/archive/v{}.tar.gz".format(
        version
    ),
    keywords=["graphs", "big-data", "signal processing", "manifold-learning",],
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
