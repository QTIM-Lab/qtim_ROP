from os import makedirs, listdir
from os.path import *
import pandas as pd
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from .segmentation.segment_unet import SegmentUnet
from .preprocessing.preprocess import preprocess
from .learning.retina_net import RetiNet, locate_config
from .utils.common import make_sub_dir
from .utils.image import find_images

LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def preprocess_images(image_files, out_dir, conf_dict, skip_segmentation=False, batch_size=100, stride=(32, 32)):

    # Segmentation
    newly_segmented, already_segmented = [], []
    prep_dir = make_sub_dir(out_dir, 'preprocessed')
    seg_dir = make_sub_dir(out_dir, 'segmentation')
    ext = '.bmp'

    if skip_segmentation:
        print("Assuming input images are already segmented")
        already_segmented = image_files  # the images provided are already segmented
    else:

        try:   # use a U-Net to segment the input images
            unet = SegmentUnet(conf_dict['vessel_directory'], seg_dir, ext=ext, stride=stride)
            newly_segmented, already_segmented, failures = unet.segment_batch(image_files, batch_size=batch_size)
        except IOError as ioe:
            print("Unable to locate segmentation model - use 'deeprop configure' to update model location")
            print(ioe)
            raise
            
    # Resizing images for inference
    prep_conf = yaml.load(open(abspath(join(dirname(__file__), 'config', 'preprocessing.yaml'))))['pipeline']
    img_names, preprocessed_arr = [], []

    for seg_image in newly_segmented + already_segmented:

        img_name = splitext(basename(seg_image))[0]
        img_names.append(img_name)
        prep_out = join(prep_dir, img_name + ext)
        prep_img = preprocess(seg_image, prep_out, prep_conf)
        preprocessed_arr.append(prep_img)

    # Reshape array - samples, channels, height, width
    preprocessed_arr = np.asarray(preprocessed_arr) #.transpose((0, 3, 1, 2))
    print(preprocessed_arr.shape)
    return preprocessed_arr, img_names


def classify(input_imgs, out_dir, conf_dict, skip_segmentation=False, batch_size=100, stride=(8, 8)):

    if any(v is None and k in ['plus_directory', 'vessel_directory'] for k, v in conf_dict.items()):
        print("Please run 'deeprop configure' to specify the models for segmentation and classification")
        exit(1)

    # Create directories
    if not isdir(out_dir):
        makedirs(out_dir)

    csv_out = join(out_dir, "DeepROP_{}.csv".format(time.strftime("%Y%m%d-%H%M%S")))

    # Identify all images
    if isdir(input_imgs):
        image_files = find_images(input_imgs)
    elif isfile(input_imgs):
        image_files = [input_imgs]
    else:
        image_files = []
        print("Please specify a valid image file or a folder of images.")
        exit(1)

    preprocessed_arr, img_names = preprocess_images(image_files, out_dir, conf_dict,
                                                    skip_segmentation=skip_segmentation,
                                                    batch_size=batch_size,
                                                    stride=stride)

    # CNN initialization
    print("Initializing classifier")
    classifier_dir = conf_dict['plus_directory']
    model_config, rf_pkl = locate_config(classifier_dir)
    cnn = RetiNet(model_config)

    print("Performing inference...")
    y_preda = cnn.predict(preprocessed_arr)
    report = generate_report(img_names, y_preda, csv_out)
    return report


def generate_report(img_names, y_preda, csv_out, y_true=None):

    cols = ["P({})".format(LABELS[i]) for i in range(0, 3)]
    df = pd.DataFrame(data=y_preda, columns=cols, index=img_names)
    df = df[['P(No)', 'P(Pre-Plus)', 'P(Plus)']]
    df['Prediction'] = [LABELS[a] for a in np.argmax(y_preda, axis=1)]

    if y_true is not None:
        df['Ground truth'] = y_true

    df['Campbell Formula'] = df['P(No)'] + (2 * df['P(Pre-Plus)']) + (3 * df['P(Plus)'])

    print(df)
    print("Results saved to: " + csv_out)
    df.to_csv(csv_out)
    return df
