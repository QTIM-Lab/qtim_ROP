from utils.common import make_sub_dir
from time import strftime
from appdirs import AppDirs
from os import makedirs, path
import yaml
from PIL import Image
import numpy as np
from sklearn.externals import joblib

from segmentation.segment_unet import SegmentUnet, segment
from preprocessing.preprocess import preprocess
from learning.retina_net import RetiNet


LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def classify(image_path, out_dir):

    conf_dict, conf_file = initialize()

    if any(v is None for v in conf_dict.values()):
        print "Please edit '{}' and specify the segmentation/classifier models to use".format(conf_file)
        exit()

    # Create output directory
    # dir_name = strftime("%Y%m%d_%H%M%S")
    working_dir = out_dir  # make_sub_dir(out_dir, dir_name)

    # Load image
    img_name = path.splitext(path.basename(image_path))[0]
    img_arr = np.asarray(Image.open(image_path))

    # Vessel segmentation
    seg_dir = make_sub_dir(working_dir, 'segmentation')
    seg_out = path.join(seg_dir, img_name + '.bmp')

    if not path.isfile(seg_out):

        try:
            unet = SegmentUnet(conf_dict['unet_directory'])

            print "Segmenting vessels"
            vessel_img = segment(img_arr, unet)
            vessel_img = (vessel_img * 255).astype(np.uint8).reshape((vessel_img.shape[0], vessel_img.shape[1]))
            Image.fromarray(vessel_img).save(seg_out)
        except IOError:
            print "Couldn't find model for segmentation - please check the config file"
            exit()

    # Classification
    classifier_dir = conf_dict['classifier_directory']

    # Pre-process image
    prep_dir = make_sub_dir(working_dir, 'preprocessed')
    prep_out = path.join(prep_dir, img_name + '.bmp')
    prep_conf = yaml.load(open(path.join(classifier_dir, 'preprocessing.yaml')))['pipeline']
    preprocess(seg_out, prep_out, prep_conf)

    # CNN initialization
    print "Loading convolutional neural network"
    cnn_name = path.basename(classifier_dir)
    cnn = RetiNet(path.join(classifier_dir, cnn_name + '.yaml'))
    cnn.set_intermediate('flatten_3')

    # Feature extraction + inference
    inf_dir =  make_sub_dir(working_dir, 'preprocessed')

    input_img = np.asarray(Image.open(prep_out))
    input_img = input_img.reshape((1, input_img.shape[2], input_img.shape[0], input_img.shape[1]))
    feature_vector = cnn.model.predict(input_img)

    random_forest = joblib.load(path.join(classifier_dir, 'rf.pkl'))
    prediction = random_forest.predict_proba(feature_vector)
    print prediction


def initialize():

    # Setup appdirs
    dirs = AppDirs("DeepROP", "QTIM", version="0.1")

    conf_dir = dirs.user_config_dir
    conf_file = path.join(conf_dir, 'config.yaml')

    if not path.isdir(conf_dir):
        makedirs(conf_dir)

    if not path.isfile(conf_file):
        config_dict = {'unet_directory': None, 'classifier_directory': None}
        with open(conf_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    return yaml.load(open(conf_file, 'r')), conf_file


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', help='Fundus image to classify', dest='image_path', required=True)
    parser.add_argument('-o', '--out-dir', help='Folder to output results', dest='out_dir', required=True)
    args = parser.parse_args()

    classify(args.image_path, args.out_dir)

