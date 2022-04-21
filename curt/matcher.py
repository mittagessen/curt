# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from itertools import zip_longest

class DummyMatcher(nn.Module):
    """
    This class computes a dummy assignment between the targets and the
    predictions of the network, i.e. returns the target indices in order to
    make the network learn the object order.
    """
    def __init__(self, *args, **kwargs):
        """Creates the matcher

        Params:
            cost_class: Ignored
            cost_curve: Ignored
        """
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_curves": Tensor of dim [batch_size, num_queries, 8] with the predicted curve coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_curves] (where num_target_curvesis the number of ground-truth
                           objects in the target) containing the class labels
                 "curves": Tensor of dim [num_target_curves, 8] containing the target curve coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_curves)
        """
        num_queries = outputs["pred_logits"].shape[1]
        sizes = [len(v["curves"]) for v in targets]
        return [(torch.arange(0, min(q, t), dtype=torch.int64),
                 torch.arange(0, min(q, t), dtype=torch.int64)) for q, t in zip_longest([num_queries], sizes, fillvalue=num_queries)]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_curve: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_curve: This is the relative weight of the L1 error of the curve coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_curve = cost_curve
        assert cost_class != 0 or cost_curve != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_curves": Tensor of dim [batch_size, num_queries, 8] with the predicted curve coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_curves] (where num_target_curvesis the number of ground-truth
                           objects in the target) containing the class labels
                 "curves": Tensor of dim [num_target_curves, 8] containing the target curve coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_curves)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_curves = outputs["pred_curves"].flatten(0, 1)  # [batch_size * num_queries, 8]

        # Also concat the target labels and curves 
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_curves = torch.cat([v["curves"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between curves 
        cost_curves = torch.cdist(out_curves, tgt_curves, p=1)

        # Final cost matrix
        C = self.cost_curve * cost_curves + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["curves"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
