import argparse
import sys
import qtim_ROP
from utils.common import find_images
from appdirs import AppDirs
from os import makedirs
from os.path import isdir, isfile, join, abspath
import yaml
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution('qtim_ROP').version
except DistributionNotFound:
    __version__ = None


class DeepROPCommands(object):
    def __init__(self):

        self.conf_dict, self.conf_file = initialize()

        parser = argparse.ArgumentParser(
            description='A set of commands for segmenting and classifying retinal images',
            usage='''deeprop <command> [<args>]

            The following commands are available:
               configure                    Configure DeepROP models to use for segmentation and classification
               segment_vessels              Perform vessel segmentation using a trained U-Net
               classify_plus                Classify plus disease in retinal images
            ''')

        parser.add_argument('command', help='Command to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def configure(self):

        parser = argparse.ArgumentParser(
            description='Update models for vessel segmentation and/or classification')

        parser.add_argument('-s', '--segmentation', help='Folder containing trained U-Net model and weights',
                            dest='unet', default=None)
        parser.add_argument('-c', '--classifier', help='Folder containing trained GoogLeNet model and weights',
                            dest='classifier', default=None)
        args = parser.parse_args(sys.argv[2:])

        if not (args.unet or args.classifier):
            print "DeepROP: no models specified."
            parser.print_usage()
            print "Current segmentation model: {unet_directory}"\
                  "\nCurrent classifier model: {classifier_directory}".format(**self.conf_dict)
            exit(1)

        self.conf_dict['unet_directory'] = abspath(args.unet)
        self.conf_dict['classifier_directory'] = abspath(args.classifier)
        yaml.dump(self.conf_dict, open(self.conf_file, 'w'), default_flow_style=False)
        print "Configuration updated!"

    def segment_vessels(self):

        parser = argparse.ArgumentParser(
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

        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument('-i', '--image-dir', help='Folder of images to classify', dest='image_dir', required=True)
        parser.add_argument('-o', '--out-dir', help='Folder to output results', dest='out_dir', required=True)
        parser.add_argument('-b', '--batch-size', help='Number of images to process at once', dest='batch_size',
                            type=int, default=10)
        args = parser.parse_args(sys.argv[2:])

        qtim_ROP.deep_rop.classify(args.image_dir, args.out_dir, self.conf_dict, batch_size=args.batch_size)


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
