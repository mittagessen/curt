"""
CURT model and criterion classes.
"""
import math
import torch
import torch.optim.lr_scheduler
import torch.nn.functional as F

from torch import nn
from pytorch_lightning import LightningModule
from typing import Tuple

from curt.util.misc import NestedTensor, nested_tensor_from_tensor_list, accuracy, inverse_sigmoid
from curt.head import CurveFormerHead, SegmentationHead, CurveHead
from curt.matcher import HungarianMatcher, DummyMatcher
from curt.cdetr_transformer import Transformer
from curt.detr_stuff import Backbone
from curt.scheduler import get_cosine_schedule_with_warmup

class CurtCurveModel(LightningModule):
    def __init__(self,
                 num_classes: int,
                 num_queries: int = 200,
                 learning_rate: float = 6e-5,
                 backbone_learning_rate: float = 6e-6,
                 weight_decay: float = 1e-4,
                 lr_drop: int = 200,
                 match_cost_class: float = 1.0,
                 match_cost_curve: float = 5.0,
                 curve_loss_coef: float = 5.0,
                 focal_alpha: float = 0.25,
                 embedding_dim: int = 256,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 dim_ff: int = 2048,
                 encoder_layers: int = 3,
                 decoder_layers: int = 3,
                 set_matcher: bool = True,
                 aux_loss: bool = False,
                 batches_per_epoch: int = -1,
                 num_epochs: int = -1):
        super().__init__()

        self.save_hyperparameters()

        self.model = Curt(num_queries=num_queries,
                          num_classes=num_classes,
                          embedding_dim=embedding_dim,
                          dropout=dropout,
                          num_heads=num_heads,
                          dim_ff=dim_ff,
                          aux_loss=aux_loss,
                          num_encoder_layers=encoder_layers,
                          num_decoder_layers=decoder_layers)

        if set_matcher:
            matcher = HungarianMatcher(cost_class=match_cost_class,
                                       cost_curve=match_cost_curve)
        else:
            matcher = DummyMatcher()

        weight_dict = {'loss_ce': 1, 'loss_curves': curve_loss_coef}

        if aux_loss:
            aux_weight_dict = {}
            for i in range(decoder_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'curves', 'cardinality']

        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                      focal_alpha=focal_alpha, losses=losses)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        if outputs['pred_curves'].isnan().any() or outputs['pred_logits'].isnan().any():
            return None
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log('loss', losses, on_epoch=False)
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
                       **loss_dict_unscaled},
                       batch_size=1)

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if 'backbone' not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if 'backbone' in n and p.requires_grad],
                "lr": self.hparams.backbone_learning_rate,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=8000,
                                                       num_training_steps=self.hparams.batches_per_epoch*self.hparams.num_epochs)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


class Curt(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,
                 num_classes: int,
                 num_queries: int,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 embedding_dim: int = 256,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 dim_ff: int = 2048,
                 pretrained: bool = True,
                 aux_loss: bool = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            num_decoder_layers: Number of decoder transformer layers.
            encoder: Encoder architecture. Might be one of `mit_b0` to `mit_b5`.
            pretrained: Load pretrained weights for encoder.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.aux_loss = aux_loss

#        self.transformer = getattr(mix_transformer, encoder)(pretrained=pretrained)
        self.backbone = Backbone(True, False)
        self.transformer = Transformer(d_model=embedding_dim,
                                       dropout=dropout,
                                       nhead=num_heads,
                                       dim_feedforward=dim_ff,
                                       num_queries=num_queries,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       normalize_before=False,
                                       return_intermediate_dec=True)

        self.input_proj = nn.Conv2d(self.backbone.num_channels, embedding_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, embedding_dim)

        self.curve_embed = CurveHead(embedding_dim, embedding_dim, 8, 3)
        # initialize curve embed
        nn.init.constant_(self.curve_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.curve_embed.layers[-1].bias.data, 0)

        self.class_embed = nn.Linear(embedding_dim, num_classes+1)

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value


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
        features, pos = self.backbone(samples.tensors)
        mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=features.shape[-2:]).to(torch.bool).squeeze(1)

        hs, references = self.transformer(self.input_proj(features), mask, self.query_embed.weight, pos)

        references_before_sigmoid = inverse_sigmoid(references)
        pred_curves = []
        for lvl in range(hs.shape[0]):
            tmp = self.curve_embed(hs[lvl])
            tmp += references_before_sigmoid
            _pred_curves = tmp.sigmoid()
            pred_curves.append(_pred_curves)
        pred_curves = torch.stack(pred_curves)

        pred_logits = self.class_embed(hs)

        out = {'pred_logits': pred_logits[-1], 'pred_curves': pred_curves[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(pred_logits, pred_curves)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_logits, pred_curves):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_curves': b}
                for a, b in zip(pred_logits[:-1], pred_curves[:-1])]


class MaskedCurtCurveModel(LightningModule):
    def __init__(self,
                 curt: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 lr_drop: int = 15,
                 match_cost_class: float = 1.0,
                 match_cost_curve: float = 5.0,
                 curve_loss_coef: float = 5.0,
                 mask_loss_coef: float = 1.0,
                 dice_loss_coef: float = 1.0,
                 eos_coef: float = 0.1):
        super().__init__()

        self.save_hyperparameters()

        self.model = MaskedCurt(num_classes=curt.num_classes, num_queries=curt.num_queries)
        self.model.load_state_dict(curt.state_dict(), strict=False)
        self.model.reinit_mask_head()

        matcher = HungarianMatcher(cost_class=match_cost_class,
                                   cost_curve=match_cost_curve)

        weight_dict = {'loss_ce': 1,
                       'loss_curves': curve_loss_coef,
                       'loss_mask': mask_loss_coef,
                       'loss_dice': dice_loss_coef}

        losses = ['labels', 'curves', 'cardinality', 'masks']

        self.criterion = SetCriterion(curt.num_classes, matcher=matcher, weight_dict=weight_dict,
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
                       **loss_dict_unscaled},
                       sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_drop)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class MaskedCurt(nn.Module):
    """ Curt with a mask on."""
    def __init__(self, num_classes: int, num_queries: int, num_decoder_layers: int = 3, encoder: str = 'mit_b0', pretrained: bool = True):
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
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.encoder = encoder
        self.transformer = getattr(mix_transformer, encoder)(pretrained=pretrained)

        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.head = CurveFormerHead(in_channels=self.transformer.embed_dims,
                                    num_queries=num_queries,
                                    num_classes=num_classes,
                                    num_decoder_layers=num_decoder_layers)

        self.mask_head = SegmentationHead(self.head)

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
               - "pred_masks":  The instance segmentation masks for each detected baseline.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.transformer(samples.tensors)
        mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=features[0].shape[-2:]).to(torch.bool).squeeze(1)
        return self.mask_head(features, mask)

    def reinit_mask_head(self):
        self.mask_head = SegmentationHead(self.head)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth curves and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and curve)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            focal_alpha: alpha in focal loss
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_curves):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_curves, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty curves
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_curves(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the cubic bezier curves: the L1 regression loss.
           targets dicts must contain the key "curves" containing a tensor of dim [nb_target_curves, 8]
           The target curves are expected in format (x0, y0, x1, y1, x2, y2,
           x3, y3), normalized by the image size.
        """
        assert 'pred_curves' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_curves = outputs['pred_curves'][idx]
        target_curves = torch.cat([t['curves'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_curves = F.l1_loss(src_curves, target_curves, reduction='none')

        losses = {}
        losses['loss_curves'] = loss_curves.sum() / num_curves
        return losses

    def loss_masks(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_curves, h, w]
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_curves),
            "loss_dice": dice_loss(src_masks, target_masks, num_curves),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_curves, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'curves': self.loss_curves,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_curves, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target curves accross all nodes, for normalization purposes
        num_curves = sum(len(t["labels"]) for t in targets)
        num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_curves = torch.clamp(num_curves, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_curves))

        # compute auxiliary losses if requested.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_curves)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

def dice_loss(inputs, targets, num_curves):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_curves


def sigmoid_focal_loss(inputs, targets, num_curves, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_curves

