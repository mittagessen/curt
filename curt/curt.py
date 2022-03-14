"""
CURT model and criterion classes.
"""
import torch
import torch.nn.functional as F

from torch import nn
from pytorch_lightning import LightningModule
from typing import Tuple

from curt.util.misc import NestedTensor, nested_tensor_from_tensor_list
from curt.mix_transformer import mit_b0
from curt.head import CurveFormerHead
from curt.matcher import HungarianMatcher


class CurtCurveModel(LightningModule):
    def __init__(self,
                 num_classes: int,
                 num_queries: int = 200,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 lr_drop: int = 200,
                 match_cost_class: float = 1.0,
                 match_cost_curve: float = 5.0,
                 curve_loss_coef: float = 5.0,
                 eos_coef: float = 0.1,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 dim_ff: int = 2048,
                 image_size: Tuple[int, int] = (1200, 800)):
        super().__init__()

        self.save_hyperparameters()

        self.model = Curt(image_size=image_size, num_queries=num_queries, num_classes=num_classes)

        matcher = HungarianMatcher(cost_class=match_cost_class,
                                   cost_curve=match_cost_curve)

        weight_dict = {'loss_ce': 1, 'loss_curves': curve_loss_coef}

        losses = ['labels', 'curves', 'cardinality']

        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                      eos_coef=eos_coef, losses=losses)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log('loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss_dict_scaled = {k: v * weight_dict[k]
                            for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                              for k, v in loss_dict.items()}
        self.log_dict({'loss': sum(loss_dict_scaled.values()),
                       'class_error': loss_dict['class_error'],
                       **loss_dict_scaled,
                       **loss_dict_unscaled})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_drop)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class Curt(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, image_size: Tuple[int, int], num_classes: int, num_queries: int, pretrained: bool = True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            pretrained: Load pretrained weights for encoder.
        """
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = mit_b0(pretrained=pretrained)

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
