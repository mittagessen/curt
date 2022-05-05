#! /usr/bin/env python

import PIL
import torch
import pathlib
import numpy as np
import pytorch_lightning as pl
import curt.transforms as tf

from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Sequence, Callable, Any, Union, Optional
from kraken.lib.xml import parse_xml
from torch.utils.data import DataLoader, random_split, Subset
from shapely.geometry import LineString

from curt.util.misc import collate_fn


class CurveDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_files: Sequence[Union[str, pathlib.Path]],
                 val_files: Optional[Sequence[Union[str, pathlib.Path]]] = None,
                 partition: Optional[float] = 0.9,
                 valid_baselines: Sequence[str] = None,
                 merge_baselines: Dict[str, Sequence[str]] = None,
                 merge_all_baselines: bool = False,
                 max_lines: int = 200,
                 batch_size: int = 1,
                 num_workers: int = 2,
                 masks: bool = False,
                 max_size: int = 1800):
        super().__init__()

        self.save_hyperparameters()

        normalize = tf.Compose([tf.ToTensor(),
                                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                tf.BezierFit()])

        scales = list(range(max_size//2, max_size, max_size//20)) + [max_size]

        self._train_transforms = tf.Compose([tf.RandomSelect(
                                                 tf.RandomResize(scales, max_size=max_size),
                                                 tf.Compose([
                                                     tf.RandomResize(scales[-3:]),
                                                     tf.RandomSizeCrop(scales[0], scales[-3]),
                                                     tf.RandomResize(scales, max_size=max_size),
                                                 ])
                                             ),
                                             tf.PhotometricDistortion(),
                                             normalize,
                                         ])

        self._val_transforms = tf.Compose([tf.RandomResize([scales[-1]], max_size=max_size),
                                           normalize])

        train_set = BaselineSet(self.hparams.train_files,
                                self._train_transforms,
                                valid_baselines=self.hparams.valid_baselines,
                                merge_baselines=self.hparams.merge_baselines,
                                max_lines=self.hparams.max_lines,
                                class_mapping=None,
                                masks=masks)

        if self.hparams.val_files:
            val_set = BaselineSet(self.hparams.val_files,
                                  self._val_transforms,
                                  valid_baselines=self.hparams.valid_baselines,
                                  merge_baselines=self.hparams.merge_baselines,
                                  max_lines=self.hparams.max_lines,
                                  class_mapping=None,
                                  masks=masks)

            train_set = Subset(train_set, range(len(train_set)))
            val_set = Subset(val_set, range(len(val_set)))
        else:
            train_len = int(len(train_set)*self.hparams.partition)
            val_len = len(train_set) - train_len

            train_set, val_set = random_split(train_set, (train_len, val_len))

        self.curve_train = train_set
        self.curve_val = val_set
        self.num_classes = train_set.dataset.num_classes

    def train_dataloader(self):
        return DataLoader(self.curve_train,
                          collate_fn=collate_fn,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.curve_val,
                          collate_fn=collate_fn,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


class BaselineSet(Dataset):
    """
    Dataset for training a baseline/region segmentation model.
    """
    def __init__(self, imgs: Sequence[str] = None,
                 im_transforms: Callable[[Any], torch.Tensor] = None,
                 valid_baselines: Sequence[str] = None,
                 merge_baselines: Dict[str, Sequence[str]] = None,
                 max_lines: int = 400,
                 class_mapping: Optional[Dict[str, int]] = None,
                 masks: bool = False):
        """
        Reads a list of image-json pairs and creates a data set.

        Args:
            imgs (list):
            target_size (tuple): Target size of the image as a (height, width) tuple.
            valid_baselines: Sequence of valid baseline identifiers. If `None`
                             all are valid.
            merge_baselines: Sequence of baseline identifiers to merge.  Note
                             that merging occurs after entities not in valid_*
                             have been discarded.
            max_lines: Threshold for maximum number of lines in input
                       documents. Pages with more lines will be discarded.
            class_mapping: Explicit mapping of type identifiers and class indices.
            masks: Returns line bounding boxes in addition to baselines.
        """
        super().__init__()
        self.im_mode = '1'
        self.targets = []
        self.mbl_dict = merge_baselines if merge_baselines is not None else {}
        self.max_lines_per_page = -1
        self.valid_baselines = valid_baselines
        self.masks = masks
        im_paths = []
        self.targets = []

        for img in imgs:
            data = parse_xml(img)
            try:
                im_size = Image.open(data['image']).size
            except (FileNotFoundError, PIL.UnidentifiedImageError):
                continue
            curves = []
            for line in data['lines']:
                if valid_baselines is None or set(line['tags'].values()).intersection(valid_baselines):
                    tags = set(line['tags'].values()).intersection(valid_baselines) if valid_baselines else line['tags'].values()
                    for tag in tags:
                        # fit baseline to cubic bezier curve
                        baseline = np.array(line['baseline'])
                        if len(baseline) < 2:
                            continue
                        if len(baseline) < 8:
                            ls = LineString(baseline)
                            baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
                        if self.masks:
                            if line['boundary']:
                                line_masks = np.array(line['boundary'])
                                curves.append({'tag': 0,
                                               'baseline': baseline,
                                               'mask': line_masks})
                        else:
                            curves.append({'tag': 0,
                                           'baseline': baseline})
            if len(curves) > max_lines:
                continue
            self.max_lines_per_page = max(self.max_lines_per_page, len(curves))
            if not len(curves):
                continue
            self.targets.append(curves)
            im_paths.append(data['image'])

        self.imgs = im_paths
        self._transforms = im_transforms
        self.num_classes = 1

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = self.targets[idx]
        return self._transforms(Image.open(im).convert('RGB'), target)

    def __len__(self):
        return len(self.imgs)
