**Curt**: Layout Analysis with Transformers
===========================================

Curt is an experimental layout analysis system based on transformers. It uses
the set prediction approach of DETR to directly predict baselines encoded as
Bézier curves and line classes. There is also an additional instance
segmentation head that predicts line bounding polygons.

Installation
------------

::

        $ pip install .


Training
--------

To train on ALTO or Page XML files on a GPU:

::

        $ curt train -d cuda train *.xml

**Warning**: Curt is resource-intensive and needs at least 30Gb of GPU memory
to train.

To train the instance segmentation head from an existing curve prediction
model: 

::

        $ curt polytrain -d cuda -i checkpoint.pth *.xml


Inference
---------

To be implemented.
