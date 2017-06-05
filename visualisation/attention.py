import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from os.path import join
from keras.preprocessing.image import img_to_array
from vis.utils import utils as vis_utils
from vis.visualization import visualize_cam
from learning.retina_net import RetiNet
from utils.common import find_images


def attention_map(model_config, image_paths, out_dir, layer_name='prob'):

    # Load model and get Keras model object
    model = RetiNet(model_config).model

    # The name of the layer to visualize
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    heatmaps = []
    for path in image_paths:

        seed_img = vis_utils.load_img(path, target_size=(224, 224))
        x = np.expand_dims(img_to_array(seed_img), axis=0)
        pred_class = np.argmax(model.predict(x))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
        heatmaps.append(heatmap)

    plt.axis('off')
    plt.imshow(vis_utils.stitch_images(heatmaps))
    plt.title('Saliency map')
    plt.savefig(join(out_dir, 'attention_maps.png'))

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='model_config', required=True)
    parser.add_argument('-i', '--image-dir', dest='image_dir', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    attention_map(args.model_config, find_images(args.image_dir), args.out_dir)
