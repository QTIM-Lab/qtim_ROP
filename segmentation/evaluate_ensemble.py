import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os.path import join, basename, dirname, abspath, isfile
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
import h5py
from segmentation import SegmentUnet
from utils.common import find_images, get_subdirs, make_sub_dir


def evaluate(models, img_dir, true_dir, out_dir):

    imgs = sorted(find_images(img_dir))
    ground_truth = [np.asarray(Image.open(img)).astype(np.bool)
                    for img in sorted(find_images(true_dir, extensions=['*.gif']))]

    all_models = sorted(get_subdirs(models))
    all_predictions = []

    for unet_dir in all_models:

        # Get model name and create folder to store segmentation results (for debugging)
        model_name = basename(unet_dir)
        print "Testing '{}'".format(model_name)
        result_dir = make_sub_dir(out_dir, 'seg_' + model_name)

        npy_file = join(out_dir, model_name + '.npy')

        if isfile(npy_file):
            print "Loading previous segmentation results"
            seg_imgs = np.load(npy_file)
        else:
            print "Performing U-Net segmentation"
            unet = SegmentUnet(unet_dir, out_dir=result_dir)
            seg_imgs = unet.segment_batch(imgs, batch_size=100)  # samples, channels, height, width
            np.save(npy_file, np.asarray(seg_imgs))

        all_predictions.append(seg_imgs)
        plot_roc_auc(seg_imgs, ground_truth, name=model_name)

    # Now run the ensemble
    ensemble_imgs = np.mean(np.asarray(all_predictions), axis=0)
    plot_roc_auc(ensemble_imgs, ground_truth, name='ensemble')

    # Make plot look pretty
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(join(out_dir, 'roc_auc.png'))


def plot_roc_auc(predictions, ground_truth, name=''):

    # Calculate ROC curve
    y_pred = np.asarray(predictions).ravel()
    y_true = np.asarray(ground_truth).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, label='{}, AUC = {:.2f}'.format(name, roc_auc))

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-e', '--ensemble', dest='ensemble', required=True)
    parser.add_argument('-i', '--images', dest='images', required=True)
    parser.add_argument('-g', '--ground-truth', dest='ground_truth', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    evaluate(args.ensemble, args.images, args.ground_truth, args.out_dir)
