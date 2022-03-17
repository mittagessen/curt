#! /usr/bin/env python

import PIL
import torch
import pathlib
import numpy as np
import pytorch_lightning as pl
import curt.transforms as tf

from PIL import Image
from collections import defaultdict
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
                 max_lines: int = 400,
                 batch_size: int = 2,
                 num_workers: int = 2):
        super().__init__()

        self.save_hyperparameters()

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        normalize = tf.Compose([tf.ToTensor(),
                                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                tf.BezierFit()])

        self._train_transforms = tf.Compose([tf.RandomHorizontalFlip(),
                                             tf.RandomSelect(
                                                 tf.RandomResize(scales, max_size=1333),
                                                 tf.Compose([
                                                     tf.RandomResize([400, 500, 600]),
                                                     tf.RandomSizeCrop(384, 600),
                                                     tf.RandomResize(scales, max_size=1333),
                                                 ])
                                             ),
                                             normalize,
                                         ])

        self._val_transforms = tf.Compose([tf.RandomResize([800], max_size=1333),
                                           normalize])

        train_set = BaselineSet(self.hparams.train_files,
                                self._train_transforms,
                                valid_baselines=self.hparams.valid_baselines,
                                merge_baselines=self.hparams.merge_baselines,
                                max_lines=self.hparams.max_lines)

        if self.hparams.val_files:
            val_set = BaselineSet(self.hparams.val_files,
                                  self._val_transforms,
                                  valid_baselines=self.hparams.valid_baselines,
                                  merge_baselines=self.hparams.merge_baselines,
                                  max_lines=self.hparams.max_lines,
                                  class_mapping=train_set.class_mapping)

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
                 class_mapping: Optional[Dict[str, int]] = None):
        """
        Reads a list of image-json pairs and creates a data set.

        Args:
            imgs (list):
            target_size (tuple): Target size of the image as a (height, width) tuple.
            valid_baselines (list): Sequence of valid baseline identifiers. If
                                    `None` all are valid.
            merge_baselines (dict): Sequence of baseline identifiers to merge.
                                    Note that merging occurs after entities not
                                    in valid_* have been discarded.
            max_lines (int): Threshold for maximum number of lines in input
                             documents. Pages with more lines will be discarded.
        """
        super().__init__()
        self.im_mode = '1'
        self.targets = []
        # n-th entry contains semantic of n-th class
        self.class_mapping = defaultdict(lambda: len(self.class_mapping) + 1) if not class_mapping else class_mapping
        self.class_stats = {'baselines': defaultdict(int)}
        self.mbl_dict = merge_baselines if merge_baselines is not None else {}
        self.max_lines_per_page = -1
        self.valid_baselines = valid_baselines
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
                        # control points normalized to image size
                        #control_pts = np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]])).flatten().tolist()
                        curves.append((self.class_mapping[self.mbl_dict.get(tag, tag)], baseline))
                        self.class_stats['baselines'][self.mbl_dict.get(tag, tag)] += 1
            if len(curves) > max_lines:
                continue
            self.max_lines_per_page = max(self.max_lines_per_page, len(curves))
            if not len(curves):
                continue
            self.targets.append(curves)
            im_paths.append(data['image'])

        self.imgs = im_paths
        self._transforms = im_transforms
        self.num_classes = max(self.class_mapping.values())

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = self.targets[idx]
        return self._transforms(Image.open(im).convert('RGB'), target)

    def __len__(self):
        return len(self.imgs)
