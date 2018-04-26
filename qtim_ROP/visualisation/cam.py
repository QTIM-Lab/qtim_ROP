import os  # set the keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
keras.backend.set_image_dim_ordering('tf')
from keras.models import load_model
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras import activations

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.backprop_modifiers import guided
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from vis.visualization import visualize_activation, visualize_saliency, visualize_cam
from vis.utils import utils

from glob import glob
import numpy as np
from PIL import Image
from os.path import dirname, join, basename, splitext
import matplotlib as mpl
import matplotlib.pyplot as plt

custom_objects = {'relu6': relu6, 'DepthwiseConv2d': DepthwiseConv2D}


def visualize_activations(model_path, layer_name="dense_6"):

    print("Visualizing activations")
    model = load_model(model_path, custom_objects=custom_objects)

    # Utility to search for layer index by name.
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model, custom_objects)

    for tv_weight in [1e-3, 1e-2, 1e-1, 1, 10]:

        for lp_norm_weight in [1e-3, 1e-2, 1e-1, 1, 10]:

            print(tv_weight)
            filter_idx = 2
            img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 255.),
                                       tv_weight=tv_weight, lp_norm_weight=lp_norm_weight)
            plt.imshow(img[..., 0], cmap='viridis')
            plt.savefig(join(dirname, 'features', model_path, 'activation_tv{}_lp_{}.png'
                             .format(tv_weight, lp_norm_weight)))


def visualize_attention(model_path, img_path, layer_name="dense_6", class_idx=2, modifier='relu'):

    print("Visualizing attention")
    model = load_model(model_path, custom_objects=custom_objects)
    try:
        layer_idx = utils.find_layer_idx(model, layer_name)
    except ValueError:
        layer_idx = -1

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model, custom_objects=custom_objects)

    for img_file in glob(join(img_path, '*')):

        if 'normal' in img_file:
            class_idx = 0 if not class_idx else class_idx
            sub_dir = 'normal'
        elif 'pre-plus' in img_file:
            class_idx = 1 if not class_idx else class_idx
            sub_dir = 'pre-plus'
        else:
            class_idx = 2 if not class_idx else class_idx
            sub_dir = 'plus'

        save_dir = join(dirname(model_path), 'features', sub_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        img = np.asarray(Image.open(img_file))
        print("Class activation map for '{}'".format(img_file))

        heatmap = visualize_cam(model, layer_idx, filter_indices=class_idx, seed_input=img, backprop_modifier=modifier)
        plt.clf()
        plt.imshow(img, cmap='gray')
        plt.imshow(heatmap, cmap='viridis', alpha=0.5)
        plt.colorbar()
        plt.savefig(join(save_dir, '{}_{}.png'.format(modifier, splitext(basename(img_file))[0])), dpi=300)


if __name__ == '__main__':

    import sys
    model_path = sys.argv[1]
    data_path = sys.argv[2]

    # visualize_activations(model_path)
    visualize_attention(model_path, data_path, modifier='relu')

