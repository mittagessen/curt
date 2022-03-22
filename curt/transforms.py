# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from scipy.special import comb
from shapely.geometry import LineString, Polygon

from shapely.ops import clip_by_rect
from skimage.draw import polygon
from skimage.color import rgb2hsv, hsv2rgb


# adapted from mmseg
class PhotometricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):

        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @classmethod
    def convert(cls, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1):
            img = rgb2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = hsv2rgb(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 1):
            img = rgb2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = hsv2rgb(img)
        return img

    def __call__(self, image, target):
        """
        Call function to perform photometric distortion on images.
        """
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        print(f'before: {image.max()}')
        # random brightness
        image = self.brightness(image)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            image = self.contrast(image)

        # random saturation
        image = self.saturation(image)

        # random hue
        image = self.hue(image)

        # random contrast
        if mode == 0:
            image = self.contrast(image)

        print(f'after: {image.max()}')

        return image, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    # top, left, height, width
    i, j, h, w = region

    curves = []
    for curve in target:
        line = clip_by_rect(LineString(curve['baseline']), j, i, j+w, i+h)
        if 'mask' in curve:
            mask  = clip_by_rect(Polygon(curve['mask']), j, i, j+w, i+h)
        if line.type == 'LineString' and line.length and ('mask' not in curve or (mask.type == 'Polygon' and mask.area > 0)):
            curves.append({'tag': curve['tag'],
                           'baseline': np.array(line.coords)-[j, i]})
            if 'mask' in curve:
                curves[-1]['mask'] = np.array(mask.boundary.coords)-[j, i]

    return cropped_image, curves

# done
def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    curves = []
    for curve in target:
        curves.append({'tag': curve['tag'],
                       'baseline': curve['baseline'] * [-1, 1] + [w, 0]})
        if 'mask' in curve:
            curves[-1]['mask'] = curve['mask'] * [-1, 1] + [w, 0]
    return flipped_image, curves

# done
def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    curves = []
    for curve in target:
        curves.append({'tag': curve['tag'],
                       'baseline': curve['baseline'] * [ratio_width, ratio_height]})
        if 'mask' in curve:
            curves[-1]['mask'] = curve['mask'] * [ratio_width, ratio_height]

    return rescaled_image, curves

# done
def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    return padded_image, target

# done
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

# done
class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

# done
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

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


class BezierFit(object):
    def __init__(self, min_points: int = 8):
        """
        Fits and normalizes a polyline to a bezier curve.

        Args:
            min_points: Minimum number of points in each line used for spline
                        fit. If the input has less points additional points
                        will be sampled.
        """
        self.min_points = min_points

    def __call__(self, image, target):
        labels = []
        curves = []
        if isinstance(image, PIL.Image.Image):
            im_size = image.size
        else:
            im_size = tuple(image.shape[1:][::-1])

        masks = None
        if len(target) and 'mask' in target[0]:
            masks = torch.zeros((len(target),) + im_size[::-1])
        for idx, line in enumerate(target):
            label, curve = line['tag'], line['baseline']
            if len(curve) < self.min_points:
                ls = LineString(curve)
                curve = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
            # control points normalized to image size
            curves.append((np.concatenate(([curve[0]], bezier_fit(curve), [curve[-1]]))/im_size).flatten().tolist())
            # create pixel map for this line
            if 'mask' in line:
                rr, cc = polygon(line['mask'][:, 1], line['mask'][:, 0], shape=im_size[::-1])
                masks[idx, rr, cc] = 1
            labels.append(label)
        ttarget = {'labels': torch.LongTensor(labels),
                   'curves': torch.Tensor(curves) if curves else torch.zeros((0, 8))}
        if masks is not None:
            ttarget['masks'] = masks
        return image, ttarget

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
