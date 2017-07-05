import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, isfile
import cv2
import numpy as np
from ..learning.retina_net import RetiNet
from ..utils.common import find_images_by_class, make_sub_dir

CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def occlusion_heatmaps(model_config, test_data, out_dir, no_imgs=None, window_size=24):

    # Load model
    model = RetiNet(model_config).model
    imgs_by_class = find_images_by_class(test_data)

    for class_, img_list in imgs_by_class.items():

        class_dir = make_sub_dir(out_dir, class_)

        no_imgs = len(img_list) if no_imgs is None else int(no_imgs)
        img_arr = []

        for img_path in img_list[:no_imgs]:

            # Load and prepare image
            img = cv2.imread(img_path)
            img = img.transpose((2, 0, 1))
            img_arr.append(img)

        # Create single array of inputs
        img_arr = np.stack(img_arr, axis=0)

        # Get raw predictions
        raw_probabilities = model.predict_on_batch(img_arr)
        raw_predictions = [np.argmax(y_pred) for y_pred in raw_probabilities]

        # Occlude overlapping windows in the image
        x_dim = img_arr.shape[2]
        y_dim = img_arr.shape[3]
        hw = window_size / 2

        hmaps_out = join(class_dir, 'heatmaps.npy')
        # debug_dir = make_sub_dir(class_dir, 'debug')

        if not isfile(hmaps_out):

            heatmaps = np.zeros(shape=(no_imgs, x_dim, y_dim))

            for x in range(0, x_dim, hw):
                for y in range(0, y_dim, hw):

                    occ_img = np.copy(img_arr)  # create copy

                    x_min, x_max = np.max([0, x-hw]), np.min([x+hw, x_dim])
                    y_min, y_max = np.max([0, y-hw]), np.min([y+hw, y_dim])

                    occ_img[:, :, x_min:x_max, y_min:y_max] = 0

                    # cv2.imwrite(join(debug_dir, '{}_{}.png'.format(x, y)), np.transpose(occ_img[0], (1, 2, 0)))

                    # Get predictions for current occluded region
                    occ_probabilities = model.predict_on_batch(occ_img)
                    print occ_probabilities

                    # Assign heatmap value as probability of class, as predicted without occlusion
                    for i, (occ_prob, raw_pred) in enumerate(zip(occ_probabilities, raw_predictions)):
                        heatmaps[i, x_min:x_max, y_min:y_max] = occ_prob[raw_pred] * 100

            np.save(hmaps_out, heatmaps)

        else:
            heatmaps = np.load(hmaps_out)

        plot_heatmaps(img_arr, heatmaps, class_dir)


def plot_heatmaps(img_arr, heatmaps, out_dir):

    for j, (img, h_map) in enumerate(zip(img_arr, heatmaps)):

        fig, ax = plt.subplots()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img, cmap='gray')
        plt.imshow(h_map, cmap=plt.cm.magma, alpha=0.7, interpolation='bilinear')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(join(out_dir, '{}.png'.format(j)), bbox_inches='tight')

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model-config', dest='model_config', help='Model config (YAML) file', required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help='Test data', required=True)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Size of occluded patch', default=24)
    parser.add_argument('-n', '--no-imgs', dest='no_imgs', help='Number of images to test with', default=None)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()

    occlusion_heatmaps(args.model_config, args.test_data, args.out_dir, no_imgs=args.no_imgs, window_size=args.window_size)
