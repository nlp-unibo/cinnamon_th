.. cinnamon-th

Cinnamon Torch Package
================================================

The torch package offers ``Component`` and related ``Configuration`` that rely on the PyTorch library.

Thus, ``cinnamon-th`` mainly provides ``Model``, ``Callback`` and ``Helper`` implementations.

===============================================
Components and Configurations
===============================================

The pytorch package defines the following ``Component`` and ``Configuration``

- Neural networks: a high-level implementation to build your torch-based neural models.
- Callbacks (e.g., early stopping)
- Framework helpers for tensorflow/cuda deterministic behaviour

===============================================
Install
===============================================

pip
   .. code-block:: bash

      pip install cinnamon-th

git
   .. code-block:: bash

      git clone https://github.com/nlp-unibo/cinnamon_th


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:
   :titlesonly:

    Model <model.rst>
    Callback <callback.rst>
    Helper <helper.rst>
    Catalog <catalog.rst>
    cinnamon-th <cinnamon_th.rst>