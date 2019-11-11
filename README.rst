==========
graphtools
==========

.. image:: https://img.shields.io/pypi/v/graphtools.svg
    :target: https://pypi.org/project/graphtools/
    :alt: Latest PyPi version
.. image:: https://anaconda.org/conda-forge/graphtools/badges/version.svg
    :target: https://anaconda.org/conda-forge/graphtools/
    :alt: Latest Conda version
.. image:: https://api.travis-ci.com/KrishnaswamyLab/graphtools.svg?branch=master
    :target: https://travis-ci.com/KrishnaswamyLab/graphtools
    :alt: Travis CI Build
.. image:: https://img.shields.io/readthedocs/graphtools.svg
    :target: https://graphtools.readthedocs.io/
    :alt: Read the Docs
.. image:: https://coveralls.io/repos/github/KrishnaswamyLab/graphtools/badge.svg?branch=master
    :target: https://coveralls.io/github/KrishnaswamyLab/graphtools?branch=master
    :alt: Coverage Status
.. image:: https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow
    :target: https://twitter.com/KrishnaswamyLab
    :alt: Twitter
.. image:: https://img.shields.io/github/stars/KrishnaswamyLab/graphtools.svg?style=social&label=Stars
    :target: https://github.com/KrishnaswamyLab/graphtools/
    :alt: GitHub stars
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

Tools for building and manipulating graphs in Python.

Installation
------------

graphtools is available on `pip`. Install by running the following in a terminal::

    pip install --user graphtools

Alternatively, graphtools can be installed using `Conda <https://conda.io/docs/>`_ (most easily obtained via the `Miniconda Python distribution <https://conda.io/miniconda.html>`_)::

    conda install -c conda-forge graphtools

Or, to install the latest version from github::

    pip install --user git+git://github.com/KrishnaswamyLab/graphtools.git

Usage example
-------------

The `graphtools.Graph` class provides an all-in-one interface for k-nearest neighbors, mutual nearest neighbors, exact (pairwise distances) and landmark graphs.

Use it as follows::

    from sklearn import datasets
    import graphtools
    digits = datasets.load_digits()
    G = graphtools.Graph(digits['data'])
    K = G.kernel
    P = G.diff_op
    G = graphtools.Graph(digits['data'], n_landmark=300)
    L = G.landmark_op

Help
----

If you have any questions or require assistance using graphtools, please contact us at https://krishnaswamylab.org/get-help
