import json
from typing import Union, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import LOSSES
from ...datasets.deepscoresv2 import get_thresholds


@LOSSES.register_module
class StatisticalLoss(nn.Module):
    """
    Loss calculation for proposals based on statistical thresholds

    Basic concept:
    Each category (area, angle, short edge, long edge, ratio between long and short edge) can have manually set
    thresholds based on compiled statistics. In every of those categories we may have a lower and a higher threshold as
    well as the standard deviation of that category from the statistics. To calculate a loss based on the threshold, we
    first subtract the measured category by the mean of the lower and higher threshold and use abs() on the result.
    This way, the region outside the two thresholds coincide. Then we subtract the distance of the mean to either
    threshold, which moves the values sitting on the edge of the threshold to zero (values within the threshold would be
    either 0 or negative). Now, we can scale the values with the std from the statistics to keep things normalized and
    cut off the negative (=accepted) values.
    """

    def __init__(self, stats_file: str, cls_out_channels: int, use_sigmoid_cls: bool,
                 target_means: List[int], target_stds: List[int]):
        super().__init__()
        self.stats = json.load(open(stats_file, mode='r'))
        self.cls_out_channels = cls_out_channels
        self.use_sigmoid_cls = use_sigmoid_cls
        self.target_means = target_means
        self.target_stds = target_stds

    def forward(self, area: Tensor, angle: Tensor, l1: Tensor, l2: Tensor, ratio: Tensor, cls: Tensor, confid: Tensor):
        """
        Calculate the loss of each proposed bbox according to the thresholds (including weighted with the confidence)
        for every available class.

        For the class loss, we compare the proposed class with the one chosen to have the lowest threshold loss,
        assuming it is the correct class. This could mean we get a more than one candidate per bbox, because some
        thresholds are the same or the values get "accepted" by multiple class thresholds. In that case, we check if the
        proposed class is intersecting with any of the candidates we calculated and if it does, we select that one for
        the final class loss calculation.
        The final class loss calculation just returns the ratio of the proposed classes matching the classes inferred
        by the threshold calculation. If there is no proposed class, we return 1.0, signalling a 100% mismatch.

        For the bbox loss, we get a boolean array indicating which entry matches the proposed class. With that we can
        just pick out the loss value from the threshold loss array.
        The final bbox loss calculation is now a mean of all the category losses' sums. If there is no proposed bboxes,
        we return 100.0 as a default loss.

        Weaknesses/things to improve:
        - We use a very hard approach on this loss calculation, which produces a rather unstable loss. Maybe something
          softer would be better.
        """
        n_preds = cls.shape[0]
        column_shape = (n_preds, 1)
        cls_ = cls.reshape(column_shape).type(torch.long) + 1
        losses = torch.zeros(column_shape, device=cls.device)
        thr_losses = torch.zeros(column_shape, device=cls.device)
        ALL_CLASSES = torch.tensor(list(map(float, self.stats.keys())), device=cls.device)
        n_classes = ALL_CLASSES.shape[0]
        row_shape = (1, n_classes)
        # For each category values and its corresponding thresholds (over all classes)
        for val, (val_min, val_max, val_std) in zip((area, angle, l1, l2, ratio), self.get_thresholds_by_classes(ALL_CLASSES)):
            # Replicate the category value for each class
            nval = val.reshape(column_shape).repeat(row_shape)
            val_min = val_min.reshape(row_shape).type(torch.float32)
            val_max = val_max.reshape(row_shape).type(torch.float32)
            val_std = val_std.reshape(row_shape).type(torch.float32)
            # Take the mean of the thresholds to be a row, aligning with the row of the replicated category values
            mean = torch.mean(torch.cat((val_min, val_max), dim=0), dim=0).reshape(row_shape)
            # Calculate the loss
            zeros = torch.zeros_like(nval, device=val.device)
            value = zeros.addcdiv(torch.abs(nval - mean) - (val_max - mean), val_std)
            if thr_losses.shape != value.shape:
                # We did not know how many classes (=rows) we were going to have, so we have to fix that for the buffer
                thr_losses = thr_losses.repeat(row_shape)
            thr_losses += value.where(value > 0.0, zeros)
        thr_losses = thr_losses * confid.reshape(column_shape)
        # The loss depends on the predicted class
        # The best class is the one where the column in the row has the lowest loss
        lowest_loss_class_indices = torch.eq(thr_losses, thr_losses.min(dim=1, keepdims=True).values)
        label_candidates = lowest_loss_class_indices * ALL_CLASSES.type(torch.long)
        # If label candidates and predicted classes intersect, chose the intersecting ones
        # (because later we use .max() to get the first candidate, which might not be the same)
        matching_candidates = (cls_.repeat(row_shape) == label_candidates)
        match_cand_idx = matching_candidates.max(dim=1).values
        # Update the candidates where the proposals match so only those remain
        label_candidates[match_cand_idx] = cls_.repeat(row_shape)[match_cand_idx]
        chosen_label = label_candidates.max(dim=1).values
        class_loss = (chosen_label != cls + 1).type(torch.float).mean() if n_preds > 0 else 1.0

        # Match all available classes with the predicted classes over all samples to get the index of the bbox losses
        bbox_loss_indx = torch.eq(ALL_CLASSES.reshape((1, n_classes)).repeat(n_preds, 1), cls_.repeat(1, n_classes) + 1)
        assert bbox_loss_indx.sum() == n_preds, "There must only be one matched class per prediction. Calculation is faulty"
        # The loss is at the position indicated by the boolean index. Grab the bbox loss from there
        bbox_loss = thr_losses[bbox_loss_indx].mean() if n_preds > 0 else 100.0
        return bbox_loss, class_loss

    def get_thresholds_by_classes(self, cls: Tensor) -> Tuple[
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor]]:
        area_min, area_max, area_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        angle_min, angle_max, angle_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        l1_min, l1_max, l1_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        l2_min, l2_max, l2_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        ratio_min, ratio_max, ratio_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)

        all_threshold_tensors = (area_min, area_max, area_std), (angle_min, angle_max, angle_std),\
                                (l1_min, l1_max, l1_std), (l2_min, l2_max, l2_std), (ratio_min, ratio_max, ratio_std)
        for i, cls_id in enumerate(cls.tolist()):
            cls_id = int(cls_id)
            if str(cls_id) not in self.stats.keys():
                # Without stats, we cannot use the std to calculate the loss
                continue
            cls_stat = self.stats[str(cls_id)]
            threshs = get_thresholds(cls_stat)
            for (key, (low, high)), (val_min, val_max, val_std) in zip(threshs.items(), all_threshold_tensors):
                if low is None or high is None:
                    continue
                val_min[i] = low
                val_max[i] = high
                val_std[i] = max(1.0, cls_stat[key]['std'])
        return all_threshold_tensors
