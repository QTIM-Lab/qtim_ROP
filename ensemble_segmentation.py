#!/usr/bin/env python

from os import listdir, sep
from os.path import join, basename, isdir
import numpy as np
from PIL import Image
from segmentation import SegmentUnet
from common import find_images, make_sub_dir


class UnetEnsemble(object):

    def __init__(self, models, out_dir, evaluate=None):

        self.models = models
        self.out_dir = out_dir
        self.evaluate = evaluate

        self.max_dir = make_sub_dir(self.out_dir, 'max')
        self.mean_dir = make_sub_dir(self.out_dir, 'mean')

    def segment_all(self, img_dir):

        result_dirs = {}
        imgs = find_images(img_dir)

        for i, model_dir in enumerate(self.models):

            print "Instantiating model #{}: {}".format(i+1, model_dir)

            result_dir = join(self.out_dir, str(i))  # create directory to store segmented images
            print "Segmented images will be written to '{}'".format(result_dir)

            model_id = basename(model_dir.rstrip(sep))  # identifier for this particular model
            result_dirs[model_id] = result_dir  # dictionary to map between IDs and results

            model = SegmentUnet(result_dir, model_dir, stride=(4, 4))  # instantiate U-Net
            model.segment_batch(imgs)  # segment images

        # self.ensemble(result_dirs)  # combine the result of the segmentations

    def ensemble(self, results):

        if self.evaluate in results.keys():
            results.pop(self.evaluate)  # don't include the evaluation data in the ensembling

        segmented_images = [find_images(x) for _, x in results.items()]

        for seg_images in zip(*segmented_images):

            im_name = basename(seg_images[0])
            seg_arr = np.asarray([np.asarray(Image.open(seg)) for seg in seg_images])

            mean_image = np.mean(seg_arr, axis=2)
            Image.fromarray(mean_image).save(join(self.mean_dir, im_name))

            max_image = np.max(seg_arr, axis=2)
            Image.fromarray(max_image).save(join(self.max_dir, im_name))

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model-dir', dest='model_dir', required=True)
    parser.add_argument('-o', '--output-dir', dest='out_dir', required=True)
    parser.add_argument('-i', '--images', dest='img_dir', required=True)

    args = parser.parse_args()

    # List of all models to ensemble
    models_list = sorted([join(args.model_dir, name) for name in listdir(args.model_dir)
                          if isdir(join(args.model_dir, name))])
    print models_list

    # Instantiate ensembler object
    ensembler = UnetEnsemble(models_list, args.out_dir, evaluate='splitAll_results')
    ensembler.segment_all(args.img_dir)  # segment images using all models and ensemble
