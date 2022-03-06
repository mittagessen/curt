#! /usr/bin/env python
"""
Produces semi-transparent neural segmenter output overlays
"""
import click

@click.command()
@click.argument('files', nargs=-1)
def cli(files):

    import sys
    import numpy as np
    import torch
    import dataset
    from PIL import Image, ImageDraw
    from os.path import splitext
    import torchvision.transforms as tf

    torch.set_num_threads(1)
    transforms = tf.Resize(800, max_size=1333)

    ds = dataset.BaselineSet(files, im_transforms=transforms)

    for idx, (im, target) in enumerate(ds):
        print(ds.imgs[idx])
        im = im.convert('RGB')
        draw = ImageDraw.Draw(im)
        samples = np.linspace(0, 1, 20)
        for line in target['curves']:
            line = (np.array(line) * (im.size * 4))
            line.resize(4, 2)
            for t in np.array(dataset.BezierCoeff(samples)).dot(line):
                draw.rectangle((t[0]-2, t[1]-2, t[0]+2, t[1]+2), fill='red')
        del draw
        im.save(splitext(ds.imgs[idx])[0] + '.overlay.png')

if __name__ == '__main__':
    cli()

