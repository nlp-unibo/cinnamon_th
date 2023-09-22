.. _helper:

``THHelper``
*************************************

The ``THHelper`` is an extension of ``Helper`` (from ``cinnamon-generic``) to handle

- cuda stochasticity
- pytorch stochasticity
- GPU visibility

All these functionality are provided within the same ``run()`` interface of ``Helper`` parent class

.. code-block:: python

    helper.run(seed=42)


The ``THHelper`` uses ``THHelperConfig`` as the default configuration template.

.. code-block:: python

    class THHelperConfig(Configuration):

        @classmethod
        def get_default(
                cls: Type[C]
        ) -> C:
            config = super().get_default()

            config.add(name='deterministic',
                       value=False,
                       type_hint=bool,
                       description='If enabled, tensorflow will run operations in deterministic mode.'
                                   'Note that this behaviour may seldom raise CUDA-related errors.')

            config.add(name='limit_gpu_visibility',
                       value=True,
                       type_hint=bool,
                       description='If enabled, it forces Tensorflow gpu visibility to the specified devices only')

            config.add(name='gpu_indexes',
                       value=[0],
                       type_hint=Iterable[int],
                       description='List of gpu indexes to make available to Tensorflow')

            config.add_condition(name='gpu_visibility',
                                 condition=lambda params: (params.limit_gpu_visibility and params.gpu_indexes is not None)
                                                          or not params.limit_gpu_visibility)

            return config

***************************
Registered configurations
***************************

The ``cinnamon-th`` package provides the following registered configurations:

- ``name='helper', tags={'default'}, namespace='th'``: the default ``THHelperConfig``.
