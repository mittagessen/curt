#! /usr/bin/env python


import torch
import pathlib
import numpy as np
import pytorch_lightning as pl

from PIL import Image
from collections import defaultdict
from torchvision import transforms as tf
from scipy.special import comb
from torch.utils.data import Dataset
from typing import Dict, Sequence, Callable, Any, Union, Optional
from kraken.lib.xml import parse_xml
from torch.utils.data import DataLoader, random_split, Subset

from util.misc import collate_fn

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

        normalize = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._train_transforms = tf.Compose([tf.Resize(800, max_size=1333),
                                             normalize])
        self._val_transforms = self._train_transforms

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


# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n,k)


def BezierCoeff(ts):
    return [[Mtk(3,t,k) for k in range(4)] for t in ts]


def bezier_fit(bl):
    x = bl[:, 0]
    y = bl[:, 1]
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(bl)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1,:]
    return medi_ctp


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
            except FileNotFoundError:
                continue
            labels = []
            curves = []
            for line in data['lines']:
                if valid_baselines is None or set(line['tags'].values()).intersection(valid_baselines):
                    tags = set(line['tags'].values()).intersection(valid_baselines) if valid_baselines else line['tags'].values()
                    for tag in tags:
                        # fit baseline to cubic bezier curve
                        baseline = np.array(line['baseline'])
                        # control points normalized to image size
                        control_pts = (np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]]))/im_size).flatten().tolist()
                        curves.append(control_pts)
                        labels.append(self.class_mapping[self.mbl_dict.get(tag, tag)])
                        self.class_stats['baselines'][self.mbl_dict.get(tag, tag)] += 1
            if len(labels) > max_lines:
                continue
            self.max_lines_per_page = max(self.max_lines_per_page, len(labels))
            if not len(labels):
                continue
            self.targets.append({'labels': torch.LongTensor(labels), 'curves': torch.Tensor(curves)})
            im_paths.append(data['image'])

        self.imgs = im_paths
        self.transforms = im_transforms
        self.num_classes = max(self.class_mapping.values())

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = self.targets[idx]
        im = self.transforms(Image.open(im).convert('RGB'))
        return (im, target)

    def __len__(self):
        return len(self.imgs)
