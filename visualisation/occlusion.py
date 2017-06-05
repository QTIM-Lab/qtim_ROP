import cv2
import numpy as np
from learning.retina_net import RetiNet
from utils.common import find_images_by_class


def occlusion_heatmaps(model_config, test_data, out_dir, window_size=12):

    imgs_by_class = find_images_by_class(test_data)

    for class_, img_list in imgs_by_class.items():

        print class_

        for img_path in img_list:

            # Load and prepare image
            img = cv2.imread(img_path)
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # Load model
            model = RetiNet(model_config).model

            x = model.predict(img)
            print x

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model-config', dest='model_config', help='Model config (YAML) file', required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help='Test data', required=True)
    parser.add_argument('-w', '--window-size', dest='window_size', help='Size of occluded patch', default=(12, 12))
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()

    occlusion_heatmaps(args.model_config, args.test_data, args.out_dir, window_size=args.window_size)
