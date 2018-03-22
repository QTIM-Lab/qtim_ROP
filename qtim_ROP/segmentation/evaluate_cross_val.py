import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from PIL import Image
from ..utils.common import find_images, make_sub_dir
from ..segmentation.segment_unet import SegmentUnet
from .evaluate_ensemble import plot_roc_auc
import seaborn as sns


def evaluate_cross_val(models, test_data, out_dir):

    fig, ax = plt.subplots()

    for i in range(0, 5):

        model_dir = join(models, 'leave_{}'.format(i))
        seg_out_dir = make_sub_dir(out_dir, 'pred_{}'.format(i))

        print("Loading model '{}'".format(model_dir))
        trained_model = SegmentUnet(model_dir, out_dir=seg_out_dir)

        # Get test images and ground truth
        test_imgs_dir = join(test_data, 'split_{}'.format(i), 'test', 'images')
        gt_dir = join(test_data, 'split_{}'.format(i), 'test', '1st_manual')
        print("Testing on data from '{}'".format(test_imgs_dir))
        print("Testing on data from '{}'".format(test_imgs_dir))

        # Segment images using model
        img_list = find_images(test_imgs_dir)
        gt_list = find_images(gt_dir)

        print(img_list)
        print(gt_list)

        y_pred = trained_model.segment_batch(img_list)
        if len(y_pred) == 0:
            print("Loading previous seg")
            y_pred = [np.asarray(Image.open(img)).astype(np.float32) / 255. for img in find_images(seg_out_dir)]

        y_true = [np.asarray(Image.open(img)).astype(np.bool) for img in gt_list]

        plot_roc_auc(y_pred, y_true, name="CV #{}".format(i))

    plt.title('ROC curves, 5-fold cross-validation')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(join(out_dir, 'roc_auc_CV.svg'))


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--models', dest='models', required=True)
    parser.add_argument('-te', '--test-data', dest='test_data', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    evaluate_cross_val(args.models, args.test_data, args.out_dir)
