from os.path import join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.common import find_images, make_sub_dir
from segmentation.segment_unet import SegmentUnet
from evaluate_ensemble import plot_roc_auc


def evaluate_cross_val(models, test_data, out_dir):

    fig, ax = plt.subplots()

    for i in range(0, 5):

        model_dir = join(models, 'leave_{}'.format(i))
        seg_out_dir = make_sub_dir(out_dir, 'pred_{}'.format(i))

        print "Loading model '{}'".format(model_dir)
        trained_model = SegmentUnet(model_dir, out_dir=seg_out_dir)

        # Get test images and ground truth
        imgs_dir = join(test_data, 'split_{}'.format(i), 'images')
        gt_dir = join(test_data, 'split_{}'.format(i), 'first_manual')
        print "Testing on data from '{}'".format(imgs_dir)

        # Segment images using model
        y_pred = trained_model.segment_batch(imgs_dir)
        y_true = [np.asarray(Image.open(img)).astype(np.bool)
                  for img in sorted(find_images(gt_dir))]

        plot_roc_auc(y_pred, y_true, name="CV #{}".format(i))

    plt.savefig(join(out_dir, 'roc_auc_CV.svg'))


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--models', dest='models', required=True)
    parser.add_argument('-te', '--test-data', dest='test_data', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    evaluate_cross_val(args.models, args.test_data, args.out_dir)