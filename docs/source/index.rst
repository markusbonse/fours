Welcome to fourS
================

This is the documentation of ``fours``, a Python package for PSF subtraction
with the 4S algorithm for exoplanet high contrast imaging (HCI).

In this documentation, we explain the how to use the package and provide some
guides to help you `get started <01_getting_started.rst>`_. We also provide the
code to `reproduce the plots <04_use_the_fours/01_general.rst>`_ in our paper.
``fours`` is fully compatible with other HCI packages like `VIP <https://vip.readthedocs.io/en/latest/>`_
or `PynPoint <https://pynpoint.readthedocs.io/en/latest/>`_. Detection limits
can be computed with `applefy <https://applefy.readthedocs.io/en/latest/>`_.

Please read the section about `Citation <05_citation.rst>`_ if you use ``fours``.

Precovery of AF Lep b
---------------------
Using the 4S algorithm, we were able to recover the planet **AF Lep b in archival
data from 2011**. In the following gallery we show the **4S** and **PCA residuals** for
this dataset.

.. raw:: html
   :file: _static/af_lep_loading.html

The planet is clearly visible in the 4S residuals irrespective of
the regularization parameter (lambda). Given the knowledge of the planet
position, we can also see it in the PCA residuals (set the components to ``65``
and combine to ``mean``).

Finding new planets with 4S
---------------------------
In the following gallery we show residual images obtained with **4S**
and the widely used **PCA** algorithm. 3 Fake planets were injected into the
data.

.. raw:: html
   :file: _static/fake_planet_residuals.html

Both algorithms have one free hyper-parameter. For PCA this is the number of
components used for the PSF model. For 4S this is the regularization parameter
``lambda``.

**Note: In the PCA residuals the signal of the fake planets sometimes co-aligns
with speckles. In practice, this can lead to a biased estimate of the planet
photometry! You can switch between the examples to see if the current PCA
residual is affected.**


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Basics

    01_getting_started

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Examples

    04_use_the_fours/01_general

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Package

    03_package/models
    03_package/utils


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: About

    05_citation
