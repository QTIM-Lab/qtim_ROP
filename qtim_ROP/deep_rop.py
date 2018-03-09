from os import makedirs, listdir
from os.path import *
import pandas as pd
import numpy as np
import yaml
import time
from visualisation.tsne import tsne
from segmentation.segment_unet import SegmentUnet
from preprocessing.preprocess import preprocess
from learning.retina_net import RetiNet, locate_config
from utils.common import make_sub_dir
from utils.common import find_images, find_images_recursive

LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def run_segmentation(image_files, out_dir, conf_dict, skip_segmentation=False, batch_size=100, stride=(32, 32), ext='.bmp'):

    # Segmentation
    newly_segmented, already_segmented, failures = [], [], []
    seg_dir = make_sub_dir(out_dir, 'segmentation')

    if skip_segmentation:
        print "Assuming input images are already segmented"
        already_segmented = image_files  # the images provided are already segmented
    else:

        try:   # use a U-Net to segment the input images
            unet = SegmentUnet(conf_dict['unet_directory'], seg_dir, ext=ext, stride=stride)
            newly_segmented, already_segmented, failures = unet.segment_batch(image_files, batch_size=batch_size)
        except IOError as ioe:
            print "Unable to locate segmentation model - use 'deeprop configure' to update model location"
            print ioe
            raise
            
    # Resizing images for inference
    return newly_segmented + already_segmented, failures


def generate_batches(image_list, out_dir, ext='.bmp', batch_size=1000):

    prep_conf = yaml.load(open(abspath(join(dirname(__file__), 'config', 'preprocessing.yaml'))))['pipeline']
    prep_dir = make_sub_dir(out_dir, 'preprocessed')
    batches = [image_list[i:i+batch_size] for i in range(0, len(image_list), batch_size)]

    for batch in batches:

        img_names, preprocessed_arr = [], []

        for seg_image in batch:

            img_name = splitext(basename(seg_image))[0]
            img_names.append(img_name)
            prep_out = join(prep_dir, img_name + ext)
            prep_img = preprocess(seg_image, prep_out, prep_conf)
            preprocessed_arr.append(prep_img)

        # Reshape array - samples, channels, height, width
        preprocessed_arr = np.asarray(preprocessed_arr).transpose((0, 3, 1, 2))
        yield preprocessed_arr, img_names


def inference(input_dir, out_dir, conf_dict, skip_segmentation=False, batch_size=100, csv_file=None,
              stride=(32, 32), features_only=False, tsne_dims=2, recursive=True):

    if any(v is None for v in conf_dict.values()):
        print "Please run 'deeprop configure' to specify the models for segmentation and classification"
        exit(1)

    # Create directories
    if not isdir(out_dir):
        makedirs(out_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_out = join(out_dir, "DeepROP_{}.csv".format(timestamp))

    # Identify all images
    if isdir(input_dir):

        if recursive:
            image_files = find_images_recursive(input_dir)
        else:
            image_files = find_images(input_dir)

        total_images = len(image_files)

        if csv_file is not None:
            csv_images = list(pd.DataFrame.from_csv(csv_file).index)
            image_files = [img for img in image_files if splitext(basename(img))[0] in csv_images]

            ignore = total_images - len(image_files)
            print "{} out of {} will be ignored, based on '{}'.".format(ignore, total_images, csv_file)

        print "{} images will be analysed.".format(len(image_files))

    elif isfile(input_dir):
        image_files = [input_dir]
    else:
        image_files = []
        print "Please specify a valid image file or folder"
        exit(1)

    if len(image_files) == 0:
        return

    # Run the segmentation part only - remove failures from image_list
    segmented_image_list, failures = run_segmentation(image_files, out_dir, conf_dict,
                                                      skip_segmentation=skip_segmentation,
                                                      batch_size=batch_size, stride=stride)

    image_files = [f for f in image_files if f not in failures]

    # Generate batches of preprocessed images
    batch_gen = generate_batches(segmented_image_list, out_dir, batch_size=5*batch_size)

    # CNN initialization
    classifier_dir = conf_dict['classifier_directory']
    model_config, rf_pkl = locate_config(classifier_dir)
    print "Initializing classifier: {}".format(model_config)
    cnn = RetiNet(model_config)

    def batch_inference():  #
        y = None
        names = []
        for img_batch, name_batch in batch_gen:
            y = cnn.predict(img_batch) if y is None else np.concatenate((y, cnn.predict(img_batch)), axis=0)
            names.extend(name_batch)
        return y, names

    if features_only:
        print "Extracting features..."
        cnn.set_intermediate('flatten_3')  # TODO add this to the YAML file at some point
        y_preda, img_names = batch_inference()
        T = tsne(y_preda, no_dims=tsne_dims)
        features_report(img_names, y_preda, T, out_dir, timestamp)
    else:
        print "Performing classification..."
        y_preda, img_names = batch_inference()
        print y_preda.shape
        print len(img_names)
        classification_report(img_names, y_preda, csv_out, full_paths=image_files)


def features_report(img_names, y_preda, T, out_dir, timestamp, y_true=None):

    features_out = join(out_dir, 'DeepROP_features_{}.csv'.format(timestamp))
    tsne_out = join(out_dir, 'DeepROP_tsne_{}.csv'.format(timestamp))

    df_features = pd.DataFrame(data=y_preda, index=img_names)
    df_tsne = pd.DataFrame(data=T, index=img_names)

    print df_features
    print df_tsne

    print "Results saved to {} and {}".format(features_out, tsne_out)
    df_features.to_csv(features_out)
    df_tsne.to_csv(tsne_out)


def classification_report(img_names, y_preda, csv_out, full_paths=None, y_true=None):

    cols = ["P({})".format(LABELS[i]) for i in range(0, 3)]
    df = pd.DataFrame(data=y_preda, columns=cols, index=img_names)
    df = df[['P(No)', 'P(Pre-Plus)', 'P(Plus)']]
    df['Prediction'] = [LABELS[a] for a in np.argmax(y_preda, axis=1)]

    if full_paths is not None:
        df['path'] = full_paths

    if y_true is not None:
        df['Ground truth'] = y_true

    df['Campbell Formula'] = df['P(No)'] + (2 * df['P(Pre-Plus)']) + (3 * df['P(Plus)'])

    print df
    print "Results saved to: " + csv_out
    df.to_csv(csv_out)
