.. _model:

``THNetwork``
*************************************

The ``THNetwork`` is an implementation of ``Network`` for PyTorch.

The following ``Network`` APIs are implemented:

- ``batch_loss``
- ``batch_train``
- ``batch_fit``
- ``batch_predict``
- ``batch_evaluate``
- ``save_model``
- ``load_model``
- ``fit``
- ``evaluate``
- ``predict``

These APIs are meant to be used with Torch-specific data structures like `torchdata.datapipes <https://pytorch.org/data/main/torchdata.datapipes.iter.html>`_.
However, **no hardcoded types** are enforced: the ``THNetwork`` works with ``FieldDict`` to ensure code flexibility.

.. note::
    The ``FieldDict`` is used to wrap data structures like ``torchdata.datapipes``

In particular, a ``FieldDict`` with the following keys **is required**:

- ``iterator``: the complete data iterator with both inputs and outputs.
- ``input_iterator``: the data iterator with only inputs.
- ``output_iterator``: the data iterator with only outputs (if any).
- ``steps``: the number of steps to take before exhausting the iterator.

For this reason, **it is always recommended** to define ad-hoc ``Processor`` components to transform input data into supported data formats.

For instance, the following ``THTextTreeProcessor`` builds a ``torch.utils.data.DataLoader`` from input data.

.. code-block:: python

    import torch as th
    from torchdata.datapipes.map import SequenceWrapper
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader

    class THTextTreeProcessor(Processor):

        def batch_data(
                self,
                data
        ):
            x, y = [], []
            for item in data:
                x.append(th.tensor(item[0], dtype=th.int32))
                y.append(item[1])

            x = pad_sequence(x, batch_first=True, padding_value=0)
            y = th.tensor(y, dtype=th.long)
            return x, y

        def process(
                self,
                data: FieldDict,
                is_training_data: bool = False
        ) -> FieldDict:
            x_th_data = SequenceWrapper(data.x)
            y_th_data = SequenceWrapper(data.y)
            th_data = x_th_data.zip(y_th_data)

            if is_training_data:
                th_data = th_data.shuffle()

            th_data = DataLoader(th_data,
                                 shuffle=is_training_data,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 collate_fn=self.batch_data)

            steps = math.ceil(len(data.nodes) / self.batch_size)

            return FieldDict({'iterator': lambda: iter(th_data),
                              'input_iterator': lambda: iter(th_data.map(lambda x, y: x)),
                              'output_iterator': lambda: iter(th_data.map(lambda x, y: y.detach().cpu().numpy())),
                              'steps': steps})