import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split

from .detr import DETR, SetCriterion
from .matcher import HungarianMatcher
from .backbone import Backbone, Joiner
from .transformer import Transformer
from .position_encoding import PositionEmbeddingSine


class CurveModel(LightningModule):
    def __init__(self,
                 num_classes: int,
                 num_queries: int = 200,
                 learning_rate: float = 1e-4,
                 learning_rate_backbone: float = 1e-5,
                 weight_decay: float = 1e-4,
                 lr_drop: int = 200,
                 aux_loss: bool = True,
                 match_cost_class: float = 1.0,
                 match_cost_curve: float = 5.0,
                 curve_loss_coef: float = 5.0,
                 eos_coef: float = 0.1,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 dim_ff: int = 2048,
                 encoder_layers: int = 6,
                 decoder_layers: int = 6,
                 backbone: str = 'resnet50'):
        super().__init__()

        self.save_hyperparameters()

        # building the backbone
        resnet = Backbone(backbone, learning_rate_backbone > 0)
        N_steps = hidden_dim // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        bb = Joiner(resnet, position_embedding)
        bb.num_channels = resnet.num_channels

        # building transformer
        transformer = Transformer(d_model=hidden_dim,
                                  dropout=dropout,
                                  nhead=num_heads,
                                  dim_feedforward=dim_ff,
                                  num_encoder_layers=encoder_layers,
                                  num_decoder_layers=decoder_layers,
                                  normalize_before=False,
                                  return_intermediate_dec=True)

        # assemble model
        self.model = DETR(bb,
                          transformer,
                          num_classes=num_classes,
                          num_queries=num_queries,
                          aux_loss=aux_loss)

        matcher = HungarianMatcher(cost_class=match_cost_class,
                                   cost_curve=match_cost_curve)

        weight_dict = {'loss_ce': 1, 'loss_curves': curve_loss_coef}

        if aux_loss:
            aux_weight_dict = {}
            for i in range(decoder_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

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
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.hparams.learning_rate_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_drop)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
