import sys
import qtim_ROP
from utils.common import find_images
from appdirs import AppDirs
from os import makedirs
from os.path import isdir, isfile, join, abspath
import yaml
from pkg_resources import get_distribution, DistributionNotFound
from argparse import ArgumentParser

try:
    __version__ = get_distribution('qtim_ROP').version
except DistributionNotFound:
    __version__ = None


class DeepROPCommands(object):
    def __init__(self):

        self.conf_dict, self.conf_file = initialize()

        parser = ArgumentParser(
            description='A set of commands for segmenting and classifying retinal images',
            usage='''deeprop <command> [<args>]

            The following commands are available:
               configure                    Configure DeepROP models to use for segmentation and classification
               segment_vessels              Perform vessel segmentation using a trained U-Net
               classify_plus                Classify plus disease in retinal images
               extract_features             Extract CNN features from retinal images
            ''')

        parser.add_argument('command', help='Command to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def configure(self):

        parser = ArgumentParser(
            description='Update models for vessel segmentation and/or classification')

        parser.add_argument('-s', '--segmentation', help='Folder containing trained U-Net model and weights',
                            dest='unet', default=None)
        parser.add_argument('-c', '--classifier', help='Folder containing trained GoogLeNet model and weights',
                            dest='classifier', default=None)
        args = parser.parse_args(sys.argv[2:])
        print args

        def print_summary():
            print "Current segmentation model: {unet_directory}"\
                  "\nCurrent classifier model: {classifier_directory}".format(**self.conf_dict)

        if not (args.unet or args.classifier):
            print "DeepROP: no models specified."
            parser.print_usage()
            print_summary()
            exit(1)

        if args.unet is not None:
            self.conf_dict['unet_directory'] = abspath(args.unet)
        if args.classifier is not None:
            self.conf_dict['classifier_directory'] = abspath(args.classifier)

        yaml.dump(self.conf_dict, open(self.conf_file, 'w'), default_flow_style=False)
        print "Configuration updated!"
        print_summary()

    def segment_vessels(self):

        parser = ArgumentParser(
            description='Perform vessel segmentation of one or more retinal images using a U-Net')

        parser.add_argument('-i', '--images', help='Directory of image files to segment', dest='images', required=True)
        parser.add_argument('-o', '--out-dir', help='Output directory', dest="out_dir", default=None)
        parser.add_argument('-u', '--unet', help='Trained U-Net directory', dest='model',
                            default=self.conf_dict['unet_directory'])
        parser.add_argument('-b', '--batch-size', help='Number of images to load at a time', default=100)

        args = parser.parse_args(sys.argv[2:])
        unet = qtim_ROP.segmentation.segment_unet.SegmentUnet(args.model, out_dir=args.out_dir)

        if isdir(args.images):
            unet.segment_batch(find_images(args.images), batch_size=args.batch_size)
        else:
            raise IOError("Please specify a valid folder of images")

    def classify_plus(self):

        parser = ArgumentParser(
            description="Classify 'plus disease' from retinal images using a deep convolutional neural network")

        parser.add_argument('-i', '--images', help='Folder of images', dest='image_dir', required=True)
        parser.add_argument('-o', '--out-dir', help='Folder to output results', dest='out_dir', required=True)
        parser.add_argument('-b', '--batch-size', help='Number of images to process at once', dest='batch_size',
                            type=int, default=10)
        parser.add_argument('--csv', help='Limit to images given in first column of CSV file', dest='csv', default=None)
        parser.add_argument('--skip-seg', help='Skip the segmentation', action='store_true', dest='skip_seg', default=False)
        parser.add_argument('--recursive', help='Search all subdirectories recursively for images', action='store_true', dest='recursive', default=True)
        args = parser.parse_args(sys.argv[2:])

        qtim_ROP.deep_rop.inference(args.image_dir, args.out_dir, self.conf_dict, skip_segmentation=args.skip_seg,
                                    batch_size=args.batch_size, csv_file=args.csv, recursive=args.recursive)

    def extract_features(self):

        parser = ArgumentParser(
            description="Extract vascular CNN features from retinal images using a deep convolutional neural network")

        parser.add_argument('-i', '--image-dir', help='Folder of images from which to extract features',
                            dest='image_dir', required=True)
        parser.add_argument('-o', '--out-dir', help='Folder to output results', dest='out_dir', required=True)
        parser.add_argument('-t', '--tsne-dims', help='Number of t-SNE latent space dimensions', dest='tsne_dims',\
                            type=int, default=2)
        parser.add_argument('-b', '--batch-size', help='Number of images to process at once', dest='batch_size',
                            type=int, default=10)
        parser.add_argument('--skip-seg', help='Skip the segmentation', action='store_true', dest='skip_seg',
                            default=False)
        args = parser.parse_args(sys.argv[2:])

        qtim_ROP.deep_rop.inference(args.image_dir, args.out_dir, self.conf_dict,
                                    skip_segmentation=args.skip_seg, batch_size=args.batch_size, features_only=True, tsne_dims=args.tsne_dims)

    def preprocess_images(self):

        parser = ArgumentParser()
        parser.add_argument('-c', '--config', dest='config', default=None)
        parser.add_argument('-o', '--out-dir', dest='out_dir', default=None)
        args = parser.parse_args(sys.argv[2:])

        pipeline = qtim_ROP.preprocessing.preprocess_cross_val.Pipeline(args.config, out_dir=args.out_dir)
        pipeline.run()


def initialize(unet=None, classifier=None):
    # Setup appdirs
    dirs = AppDirs("DeepROP", "QTIM", version=__version__)
    conf_dir = dirs.user_config_dir
    conf_file = join(conf_dir, 'config.yaml')

    if not isdir(conf_dir):
        makedirs(conf_dir)

    if not isfile(conf_file):
        config_dict = {'unet_directory': unet, 'classifier_directory': classifier}

        with open(conf_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    return yaml.load(open(conf_file, 'r')), conf_file


def main():
    DeepROPCommands()
