#!/usr/bin/env python

from os import listdir, sep
from os.path import join, basename, isfile
import numpy as np
from PIL import Image
from .segment_unet import SegmentUnet, segment
from utils.common import find_images, get_subdirs


class UnetEnsemble(object):

    def __init__(self, model_dirs):

        self.model_dirs = model_dirs
        self._load_models()

    def _load_models(self):

        self.models = []
        for i, model_dir in enumerate(self.model_dirs):
            self.models.append(SegmentUnet(model_dir, stride=(4, 4)))

    def segment_one(self, img):

        predictions = []
        for unet in self.models:

            seg = segment(img, unet)
            predictions.append(seg)

        return np.mean(np.dstack(predictions), axis=2)

    def segment_batch(self, img_list, batch_size=100):

        all_seg = []
        for unet in self.models:
            seg_imgs = unet.segment_batch(img_list, batch_size=batch_size)
            all_seg.append(seg_imgs)

        return all_seg


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model-dir', dest='model_dir', required=True)
    parser.add_argument('-o', '--output-dir', dest='out_dir', required=True)
    parser.add_argument('-i', '--images', dest='img_dir', required=True)

    args = parser.parse_args()

    # List of all models to ensemble
    models_list = sorted(get_subdirs(args.model_dir))

    # Instantiate ensembler object
    ensemble = UnetEnsemble(models_list)
    seg = ensemble.segment_batch(find_images(args.img_dir))
