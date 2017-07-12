from os import makedirs
from os.path import *
from PIL import Image
import cv2
from scipy.misc import imresize
import numpy as np
import yaml
from segmentation.segment_unet import SegmentUnet, segment
from preprocessing.preprocess import preprocess
from learning.retina_net import RetiNet, locate_config
from utils.common import make_sub_dir

LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def classify(image_path, out_dir, conf_dict):

    if any(v is None for v in conf_dict.values()):
        print "Please run 'deeprop configure' to specify the models for segmentation and classification"
        exit()

    # Create output directory
    if not isdir(out_dir):
        makedirs(out_dir)

    # Load image
    img_name = splitext(basename(image_path))[0]
    img_arr = np.asarray(cv2.imread(image_path))

    # Resize to 640 x 480
    img_arr = imresize(img_arr, (480, 640), interp='bicubic')

    # Vessel segmentation
    seg_dir = make_sub_dir(out_dir, 'segmentation')
    seg_out = join(seg_dir, img_name + '.bmp')

    if not isfile(seg_out):

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
    prep_dir = make_sub_dir(out_dir, 'preprocessed')
    prep_out = join(prep_dir, img_name + '.bmp')
    prep_conf = yaml.load(open(abspath(join(dirname(__file__), 'config', 'preprocessing.yaml'))))['pipeline']
    preprocess(seg_out, prep_out, prep_conf)

    # CNN initialization
    print "Initializing classifier network"
    model_config, rf_pkl = locate_config(classifier_dir)
    cnn = RetiNet(model_config)

    input_img = cv2.imread(prep_out)
    input_img = np.expand_dims(input_img.transpose((2, 0, 1)), axis=0)  # channels first
    y_preda = cnn.predict(input_img)[0]
    arg_max = np.argmax(y_preda)

    print "\n### {} classified as '{}' with {:.1f} % probability ###"\
        .format(basename(image_path), LABELS[arg_max], y_preda[arg_max] * 100)
