#! /usr/bin/env python
import glob
import click
import os.path
import argparse
import datetime
import random
import time
from pathlib import Path

import logging
from rich.logging import RichHandler

import numpy as np
import torch

from pytorch_lightning import Trainer

from models import CurveModel
from dataset import CurveDataModule


def set_logger(logger=None, level=logging.ERROR):
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.setLevel(level)


logging.captureWarnings(True)
logger = logging.getLogger()


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        for entry in manifest.readlines():
            im_p = entry.rstrip('\r\n')
            if os.path.isfile(im_p):
                images.append(im_p)
            else:
                logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
    return images


def _validate_merging(ctx, param, value):
    """
    Maps baseline/region merging to a dict of merge structures.
    """
    if not value:
        return None
    merge_dict = {}
    try:
        for m in value:
            k, v = m.split(':')
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict


@click.group()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
def cli(ctx, verbose, seed):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    ctx.meta['verbose'] = verbose
    set_logger(logger, level=30 - min(10 * verbose, 20))


@cli.command('train')
@click.pass_context
@click.option('-lr', '--learning-rate', default=1e-4, help='Learning rate')
@click.option('-lr-bb', '--learning-rate-backbone', default=1e-5, help='Backbone learning rate')
@click.option('-B', '--batch-size', default=2, help='Batch size')
@click.option('-w', '--weight-decay', default=1e-4, help='Weight decay in optimizer')
@click.option('-N', '--epochs', default=300, help='Number of epochs to train for')
@click.option('-F', '--freq', show_default=True, default=1.0, type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-lr-drop', '--lr-drop', default=200, help='Reduction factor of learning rate over time')
@click.option('--clip-max-norm', default=0.1, help='gradient clipping max norm')
@click.option('--train-mask-head/--no-train-mask-head', default=False, help='Flag to enable/disable training of the mask head. Pretrained weights must be loaded for this feature')
@click.option('--backbone', default='resnet50', type=click.Choice(['resnet18', 'resnet34', 'resnet50']), help='Type of network to use for feature extractor')
@click.option('-el', '--encoder-layers', default=6, help='Number of encoder layers in the transformer')
@click.option('-dl', '--decoder-layers', default=6, help='Number of decoder layers in the transformer')
@click.option('-dff', '--dim-ff', default=2048, help='Intermediate size of the feedforward layers in the transformer block')
@click.option('-hdd', '--hidden-dim', default=256, help='Size of the embeddings (dimension of the transformer')
@click.option('--dropout', default=0.1, help='Dropout applied in the transformer')
@click.option('-nh', '--num-heads', default=8, help="Number of attention heads inside the transformer's attentions")
@click.option('-nq', '--num-queries', default=500, help='Number of query slots (#lines + #regions detectable in an image)')
@click.option('--aux-loss/--no-aux-loss', default=True, help='Flag for auxiliary decoding losses (loss at each layer)')
@click.option('--match-cost-class', default=1.0, help='Class coefficient in the matching cost')
@click.option('--match-cost-curve', default=5.0, help='L1 curve coefficient in the matching cost')
@click.option('--curve-loss-coef', default=5.0, help='L1 curve coefficient in the loss')
@click.option('--eos-coef', default=0.1, help='Relative classification weight of the no-object class')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--augment/--no-augment',
              show_default=True,
              default=False,
              help='Enable image augmentation')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('--workers', show_default=True, default=2, help='Number of data loader workers.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, learning_rate, learning_rate_backbone, batch_size, weight_decay,
          epochs, freq, lr_drop, clip_max_norm, train_mask_head, backbone,
          encoder_layers, decoder_layers, dim_ff, hidden_dim, dropout,
          num_heads, num_queries, aux_loss, match_cost_class, match_cost_curve,
          curve_loss_coef, eos_coef, load, output, partition, training_files,
          evaluation_files, augment, valid_baselines, merge_baselines, workers,
          device, ground_truth):

    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if device == 'cpu':
        device = None
    elif device.startswith('cuda'):
        device = [int(device.split(':')[-1])]

    if freq > 1:
        val_check_interval = {'check_val_every_n_epoch': int(freq)}
    else:
        val_check_interval = {'val_check_interval': freq}

    if not valid_baselines:
        valid_baselines = None

    data_module = CurveDataModule(train_files=ground_truth,
                                  val_files=evaluation_files,
                                  partition=partition,
                                  valid_baselines=valid_baselines,
                                  merge_baselines=merge_baselines,
                                  max_lines=num_queries,
                                  batch_size=batch_size,
                                  num_workers=workers)

    if load:
        model = CurveModel.load_from_checkpoint(load)
    else:
        model = CurveModel(data_module.num_classes+1,
                           num_queries=num_queries,
                           learning_rate=learning_rate,
                           learning_rate_backbone=learning_rate_backbone,
                           weight_decay=weight_decay,
                           lr_drop=lr_drop,
                           aux_loss=aux_loss,
                           match_cost_class=match_cost_class,
                           match_cost_curve=match_cost_curve,
                           curve_loss_coef=curve_loss_coef,
                           eos_coef=eos_coef,
                           hidden_dim=hidden_dim,
                           dropout=dropout,
                           num_heads=num_heads,
                           dim_ff=dim_ff,
                           encoder_layers=encoder_layers,
                           decoder_layers=decoder_layers,
                           backbone=backbone)

    trainer = Trainer(default_root_dir=output, gradient_clip_val=clip_max_norm, **val_check_interval)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    cli()
