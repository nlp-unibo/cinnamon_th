import os
from typing import Optional, Any

import torch
import torch as th

from cinnamon_generic.components.helper import Helper


class THHelper(Helper):
    """
    Torch specific backend helper.
    `THHelper` controls deterministic behaviour and gpu visibility.
    """

    def set_seed(
            self,
            seed: int
    ):
        super().set_seed(seed=seed)
        th.manual_seed(seed=seed)

        if self.deterministic:
            torch.use_deterministic_algorithms(True)

    def clear(
            self
    ):
        pass

    def limit_gpu_usage(self):
        # avoid other GPUs
        if self.limit_gpu_visibility:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(idx) for idx in self.gpu_indexes])

    def run(
            self,
            seed: Optional[int] = None
    ) -> Any:
        self.set_seed(seed=seed)
        self.limit_gpu_usage()
