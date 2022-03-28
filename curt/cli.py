#! /usr/bin/env python
import glob
import time
import torch
import click
import os.path
import random
import logging
import pathlib
import datetime
import numpy as np
import torchvision.transforms as tf

from PIL import Image
from pathlib import Path
from rich.logging import RichHandler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from curt.models import CurtCurveModel, MaskedCurtCurveModel
from curt.dataset import CurveDataModule
from curt.progress import KrakenTrainProgressBar
from curt.util.misc import NestedTensor


def set_logger(logger=None, level=logging.ERROR):
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.setLevel(level)

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2

logging.captureWarnings(True)
logger = logging.getLogger()

torch.multiprocessing.set_sharing_strategy('file_system')

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

@cli.command('polytrain')
@click.pass_context
@click.option('-lr', '--learning-rate', default=1e-4, help='Learning rate')
@click.option('-B', '--batch-size', default=2, help='Batch size')
@click.option('-w', '--weight-decay', default=1e-4, help='Weight decay in optimizer')
@click.option('-N', '--epochs', default=25, help='Number of epochs to train for')
@click.option('-F', '--freq', show_default=True, default=1.0, type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-lr-drop', '--lr-drop', default=15, help='Reduction factor of learning rate over time')
@click.option('--clip-max-norm', default=0.1, help='gradient clipping max norm')
@click.option('--dropout', default=0.1, help='Dropout applied in the transformer')
@click.option('--match-cost-class', default=1.0, help='Class coefficient in the matching cost')
@click.option('--match-cost-curve', default=5.0, help='L1 curve coefficient in the matching cost')
@click.option('--curve-loss-coef', default=5.0, help='L1 curve coefficient in the loss')
@click.option('--eos-coef', default=0.1, help='Relative classification weight of the no-object class')
@click.option('--mask-loss-coef', default=1.0, help='Mask loss coefficient')
@click.option('--dice-loss-coef', default=1.0, help='Mask dice loss coefficient')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='curt_model', help='Pytorch lightning output directory')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('--merge-all-baselines/--no-merge-baselines',
              show_default=True,
              default=False,
              help='Merge all baseline types into `default`')
@click.option('--workers', show_default=True, default=2, help='Number of data loader workers.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def polytrain(ctx, learning_rate, batch_size, weight_decay, epochs, freq, lr_drop,
        clip_max_norm, dropout, match_cost_class, match_cost_curve,
        curve_loss_coef, eos_coef, mask_loss_coef, dice_loss_coef, load,
        output, partition, training_files, evaluation_files, valid_baselines,
        merge_baselines, merge_all_baselines, workers, device, ground_truth):

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



    if load:
        curt_model = CurtCurveModel.load_from_checkpoint(load).model
        model = MaskedCurtCurveModel(curt_model,
                                     learning_rate=learning_rate,
                                     weight_decay=weight_decay,
                                     lr_drop=lr_drop,
                                     match_cost_class=match_cost_class,
                                     match_cost_curve=match_cost_curve,
                                     curve_loss_coef=curve_loss_coef,
                                     mask_loss_coef=mask_loss_coef,
                                     dice_loss_coef=dice_loss_coef,
                                     eos_coef=eos_coef)
    else:
        raise click.UsageError('No pretrained weights given for mask head training.')

    data_module = CurveDataModule(train_files=ground_truth,
                                  val_files=evaluation_files,
                                  partition=partition,
                                  valid_baselines=valid_baselines,
                                  merge_baselines=merge_baselines,
                                  merge_all_baselines=merge_all_baselines,
                                  max_lines=curt_model.num_queries,
                                  batch_size=batch_size,
                                  num_workers=workers,
                                  masks=True)

    click.echo('Line types:')
    for k, v in data_module.curve_train.dataset.class_mapping.items():
        click.echo(f'{k}\t{v}')

    checkpoint_cb = ModelCheckpoint(monitor='loss', save_top_k=5, mode='min')

    trainer = Trainer(default_root_dir=output,
                      gradient_clip_val=clip_max_norm,
                      max_epochs=epochs,
                      gpus=device,
                      callbacks=[KrakenTrainProgressBar(), checkpoint_cb, StochasticWeightAveraging()],
                      **val_check_interval)

    trainer.fit(model, data_module)



@cli.command('train')
@click.pass_context
@click.option('-lr', '--learning-rate', default=1e-4, help='Learning rate')
@click.option('-B', '--batch-size', default=2, help='Batch size')
@click.option('-w', '--weight-decay', default=1e-4, help='Weight decay in optimizer')
@click.option('-N', '--epochs', default=300, help='Number of epochs to train for')
@click.option('-F', '--freq', show_default=True, default=1.0, type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-lr-drop', '--lr-drop', default=200, help='Reduction factor of learning rate over time')
@click.option('--clip-max-norm', default=0.1, help='gradient clipping max norm')
@click.option('--encoder', default='mit_b0', type=click.Choice(['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']), help='Encoding max transformers architecture')
@click.option('-dl', '--decoder-layers', default=3, help='Number of decoder layers in the transformer')
@click.option('-dff', '--dim-ff', default=2048, help='Intermediate size of the feedforward layers in the transformer block')
@click.option('-hdd', '--hidden-dim', default=256, help='Size of the embeddings (dimension of the transformer')
@click.option('--dropout', default=0.1, help='Dropout applied in the transformer')
@click.option('-nh', '--num-heads', default=8, help="Number of attention heads inside the transformer's attentions")
@click.option('-nq', '--num-queries', default=500, help='Number of query slots (#lines + #regions detectable in an image)')
@click.option('--match-cost-class', default=1.0, help='Class coefficient in the matching cost')
@click.option('--match-cost-curve', default=5.0, help='L1 curve coefficient in the matching cost')
@click.option('--curve-loss-coef', default=5.0, help='L1 curve coefficient in the loss')
@click.option('--eos-coef', default=0.1, help='Relative classification weight of the no-object class')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='curt_model', help='Pytorch lightning output directory')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('--merge-all-baselines/--no-merge-baselines',
              show_default=True,
              default=False,
              help='Merge all baseline types into `default`')
@click.option('--workers', show_default=True, default=2, help='Number of data loader workers.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, learning_rate, batch_size, weight_decay, epochs, freq, lr_drop,
          clip_max_norm, encoder, decoder_layers, dim_ff, hidden_dim,
          dropout, num_heads, num_queries, match_cost_class,
          match_cost_curve, curve_loss_coef, eos_coef, load, output, partition,
          training_files, evaluation_files, valid_baselines, merge_baselines,
          merge_all_baselines, workers, device, ground_truth):

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
                                  merge_all_baselines=merge_all_baselines,
                                  max_lines=num_queries,
                                  batch_size=batch_size,
                                  num_workers=workers)

    click.echo('Line types:')
    for k, v in data_module.curve_train.dataset.class_mapping.items():
        click.echo(f'{k}\t{v}')

    if load:
        model = CurtCurveModel.load_from_checkpoint(load)
    else:
        model = CurtCurveModel(data_module.num_classes+1,
                               num_queries=num_queries,
                               learning_rate=learning_rate,
                               weight_decay=weight_decay,
                               lr_drop=lr_drop,
                               match_cost_class=match_cost_class,
                               match_cost_curve=match_cost_curve,
                               curve_loss_coef=curve_loss_coef,
                               eos_coef=eos_coef,
                               hidden_dim=hidden_dim,
                               dropout=dropout,
                               num_heads=num_heads,
                               dim_ff=dim_ff,
                               encoder=encoder,
                               decoder_layers=decoder_layers)

    checkpoint_cb = ModelCheckpoint(monitor='loss', save_top_k=5, mode='min')

    trainer = Trainer(default_root_dir=output,
                      gradient_clip_val=clip_max_norm,
                      max_epochs=epochs,
                      gpus=device,
                      callbacks=[KrakenTrainProgressBar(), checkpoint_cb, StochasticWeightAveraging()],
                      **val_check_interval)

    trainer.fit(model, data_module)


@cli.command('pred')
@click.pass_context
@click.option('-i', '--load', help='Input model')
@click.option('-o', '--suffix', default='.overlay.png', show_default=True, help='Suffix for output files')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.argument('input_files', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def pred(ctx, load, suffix, device, input_files):

    curt_model = CurtCurveModel.load_from_checkpoint(load).model
    curt_model = curt_model.to(device)

    transforms = tf.Compose([tf.Resize(800),
                             tf.ToTensor(),
                             tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    for file in input_files:
        file = pathlib.Path(file)
        with open(file, 'rb') as fp:
            im = Image.open(file)
            with open(file.with_suffix(suffix), 'wb') as fo:
                with torch.no_grad():
                    i = transforms(im).to(device).unsqueeze(0)
                    mask = torch.zeros((1,) + i.shape[2:], device=device)
                    i = NestedTensor(i, mask)
                    o = curt_model(i)
                curves = o['pred_curves']

if __name__ == '__main__':
    cli()
