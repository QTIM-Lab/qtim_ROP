from os.path import join, isfile, splitext
import cv2
import numpy as np
from qtim_ROP.learning.retina_net import RetiNet
from qtim_ROP.utils.common import dict_reverse, make_sub_dir
from qtim_ROP.utils.image import imgs_by_class_to_th_array
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap


CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def occlusion_heatmaps(model_config, test_data, out_dir, no_imgs=None, window_size=24):

    # Load model
    model = RetiNet(model_config).model
    img_names, img_arr, y_true = imgs_by_class_to_th_array(test_data, dict_reverse(CLASSES))
    no_imgs = len(img_names) if no_imgs is None else no_imgs

    # Get raw predictions
    raw_probabilities = model.predict_on_batch(img_arr)
    raw_predictions = [np.argmax(y_pred) for y_pred in raw_probabilities]

    # Occlude overlapping windows in the image
    x_dim = img_arr.shape[2]
    y_dim = img_arr.shape[3]
    hw = window_size / 2

    hmaps_out = join(out_dir, 'heatmaps.npy')

    csv_data = []
    titles = []
    for i in range(0, len(img_names)):

        rsd = CLASSES[y_true[i]]
        pred = CLASSES[raw_predictions[i]]
        prob = raw_probabilities[i, raw_predictions[i]]

        titles.append('RSD: {}, Prediction: {} ({:.2f})'.format(rsd, pred, prob))
        csv_data.append({'Name': img_names[i], 'RSD': rsd, 'Prediction': pred, 'Probability': prob})
        pd.DataFrame(csv_data).set_index('Name').to_csv(join(out_dir, 'occlusion_data.csv'))

    if not isfile(hmaps_out):

        heatmaps = np.zeros(shape=(no_imgs, x_dim, y_dim))

        for x in range(0, x_dim):
            for y in range(0, y_dim):

                occ_img = np.copy(img_arr)  # create copy

                x_min, x_max = np.max([0, x-hw]), np.min([x+hw, x_dim])
                y_min, y_max = np.max([0, y-hw]), np.min([y+hw, y_dim])

                occ_img[:, :, x_min:x_max, y_min:y_max] = 0

                # cv2.imwrite(join(debug_dir, '{}_{}.png'.format(x, y)), np.transpose(occ_img[0], (1, 2, 0)))

                # Get predictions for current occluded region
                occ_probabilities = model.predict_on_batch(occ_img)
                # print occ_probabilities

                # Assign heatmap value as probability of class, as predicted without occlusion
                for i, (occ_prob, raw_pred, raw_prob) in enumerate(zip(occ_probabilities, raw_predictions, raw_probabilities)):
                    heatmaps[i, x, y] = raw_prob[raw_pred] - occ_prob[raw_pred]

        np.save(hmaps_out, heatmaps)

    else:
        heatmaps = np.load(hmaps_out)

    plot_heatmaps(img_arr, img_names, titles, heatmaps, y_true, out_dir)


def plot_heatmaps(img_arr, img_names, titles, heatmaps, labels, out_dir):

    # construct cmap
    pal = sns.diverging_palette(240, 10, n=30, center="dark")
    my_cmap = ListedColormap(sns.color_palette(pal).as_hex())

    min_val, max_val = np.min(heatmaps), np.max(heatmaps)

    for j, (img, img_name, h_map, title, y) in enumerate(zip(img_arr, img_names, heatmaps, titles, labels)):

        fig, ax = plt.subplots()
        img = np.transpose(img, (1, 2, 0))
        plt.clf()
        plt.imshow(img, cmap='Greys', interpolation='bicubic')
        plt.imshow(h_map, cmap=my_cmap, alpha=0.7, interpolation='nearest') #, vmin=-.05, vmax=.05)
        plt.colorbar()
        plt.axis('off')
        plt.title(title)
        class_name = CLASSES[y]
        class_dir = make_sub_dir(out_dir, class_name)
        plt.savefig(join(class_dir, img_name), bbox_inches='tight', dpi=300)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model-config', dest='model_config', help='Model config (YAML) file', required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help='Test data', required=True)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Size of occluded patch', type=int, default=24)
    parser.add_argument('-n', '--no-imgs', dest='no_imgs', help='Number of images to test with', default=None, type=int)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()

    occlusion_heatmaps(args.model_config, args.test_data, args.out_dir, no_imgs=args.no_imgs, window_size=args.window_size)
