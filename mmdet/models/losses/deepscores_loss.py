import json
from abc import ABC
from pathlib import Path
from typing import Union

import torch.nn as nn
from torch import Tensor

from ..registry import LOSSES

class StatisticalLoss(nn.Module, ABC):

    def __init__(self, stats: dict):
        super().__init__()
        self.stats = stats

    def forward(self, pred: Tensor, target: Tensor, weight: Tensor = None, avg_factor: Tensor = None,
                reduction_override: Tensor = None, **kwargs):
        raise NotImplementedError()


@LOSSES.register_module
class StatisticalBBoxLoss(StatisticalLoss):
    def forward(self, pred: Tensor, target: Tensor, weight: Tensor = None, avg_factor: Tensor = None,
                reduction_override: Tensor = None, **kwargs):
        pass


@LOSSES.register_module
class StatisticalClassLoss(StatisticalLoss):
    def forward(self, pred: Tensor, target: Tensor, weight: Tensor = None, avg_factor: Tensor = None,
                reduction_override: Tensor = None, **kwargs):
        pass