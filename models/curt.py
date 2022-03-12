"""
CURT model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


from .mix_transformer import MixVisionTransformer, mit_b0
from .head import CurveFormerHead


class Curt(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, image_size, num_classes, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = mit_b0()
        self.head = CurveFormerHead(in_channels=self.transformer.embed_dims,
                                    num_queries=num_queries,
                                    num_classes=num_classes,
                                    feature_map_size=(self.image_size[0]//4, self.image_size[1]//4))

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_curves": The normalized curve control points for all queries, represented as
                                (x0, y0, x1, y1, x2, y2, x3, y3). These values are normalized in [0, 1],
                                relative to the size of each individual image (disregarding possible padding).
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.transformer(samples.tensors)
        mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=features[0].shape[-2:]).to(torch.bool).squeeze(1)
        return self.head(features, mask)
