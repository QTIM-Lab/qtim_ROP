import sys
import qtim_ROP
from .utils.common import find_images
from appdirs import AppDirs
from os import makedirs
from os.path import isdir, isfile, join, abspath
import yaml
from pkg_resources import get_distribution, DistributionNotFound
from argparse import ArgumentParser
from .quality_assurance import QualityAssurance

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
               configure                    Configure DeepROP models to use for segmentation, classification, and QA
               classify_plus                Classify plus disease in retinal images
               segment_vessels              Perform vessel segmentation using a trained U-Net
               segment_optic_disc           Perform optic disc segmentation using a trained U-Net
               quality_assurance            Assess image quality using multiple trained models
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

        parser.add_argument('-p', '--plus', help='Folder containing model for plus disease classification',
                            dest='plus', default=None)
        parser.add_argument('-v', '--vessels', help='Folder containing model/weights trained for vessel segmentation',
                            dest='vessels', default=None)
        parser.add_argument('-o', '--optic', help='Folder containing model/weights trained for optic disc segmentation',
                            dest='optic', default=None)
        parser.add_argument('-q', '--quality', help='Folder containing model/weights trained for quality estimation',
                            dest='quality', default=None)
        parser.add_argument('-g', '--gpu', help='Numerical identifier of which GPU to use (0-N, or -1 for CPU)',
                            dest='gpu', default=None)
        args = parser.parse_args(sys.argv[2:])

        def print_summary():
            print("Current plus disease model: {plus_directory}"
                  "\nCurrent vessel segmentation model: {vessel_directory}"
                  "\nCurrent optic disc segmentation model: {optic_directory}"
                  "\nCurrent quality model: {quality_directory}"
                  "\nGPU (-1 for CPU): {gpu}".format(**self.conf_dict))

        if all([v is None for k, v in args.__dict__.items()]):
            print("DeepROP: no models specified.")
            parser.print_usage()
            print_summary()
            exit(1)

        if args.plus is not None:
            self.conf_dict['plus_directory'] = abspath(args.plus)
        if args.vessels is not None:
            self.conf_dict['vessel_directory'] = abspath(args.vessels)
        if args.optic is not None:
            self.conf_dict['optic_directory'] = abspath(args.optic)
        if args.quality is not None:
            self.conf_dict['quality_directory'] = abspath(args.quality)
        if args.gpu is not None:
            self.conf_dict['gpu'] = str(args.gpu)

        yaml.dump(self.conf_dict, open(self.conf_file, 'w'), default_flow_style=False)
        print("Configuration updated!")
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

    def quality_assurance(self):

        parser = ArgumentParser(
            description='Run quality assurance models to verify the integrity of provided data.')

        parser.add_argument('-i', '--images', help='Directory of image files to assess', dest='images', required=True)
        parser.add_argument('-o', '--out-dir', help='Output directory', dest="out_dir", required=True)
        parser.add_argument('-c', '--config', help='Configuration file specifying models to use', dest='config',
                            default=self.conf_dict)
        parser.add_argument('-b', '--batch-size', help='Number of images to assess at a time', dest='batch_size', type=int, default=50)
        parser.add_argument('-r', '--recursive', help='Option to search directory recursively', dest='recursive', action='store_true', default=False)

        args = parser.parse_args(sys.argv[2:])
        qa = QualityAssurance(args.images, args.config, out_dir=args.out_dir, batch_size=args.batch_size, recursive=args.recursive)
        return qa.run()

    def classify_plus(self):

        parser = ArgumentParser()
        parser.add_argument('-i', '--image-dir', help='Folder of images to classify', dest='image_dir', required=True)
        parser.add_argument('-o', '--out-dir', help='Folder to output results', dest='out_dir', required=True)
        parser.add_argument('-b', '--batch-size', help='Number of images to process at once', dest='batch_size',
                            type=int, default=10)
        parser.add_argument('--skip-seg', help='Skip the segmentation', action='store_true', dest='skip_seg', default=False)
        args = parser.parse_args(sys.argv[2:])

        qtim_ROP.deep_rop.classify(args.image_dir, args.out_dir, self.conf_dict,
                                   skip_segmentation=args.skip_seg, batch_size=args.batch_size)

    def run(self):
        """
        Run inference for plus disease with additional quality control measures
        :return: Spreadsheet listing images
        """

        parser = ArgumentParser()
        parser.add_argument('-i', '--image-dir', help='Folder of images to classify', dest='image_dir', required=True)
        parser.add_argument('-o', '--out-dir', help='Folder to output results', dest='out_dir', required=True)
        parser.add_argument('-b', '--batch-size', help='Number of images to process at once', dest='batch_size',
                            type=int, default=10)
        parser.add_argument('--skip-seg', help='Skip the segmentation', action='store_true', dest='skip_seg',
                            default=False)
        args = parser.parse_args(sys.argv[2:])

        qa = QualityAssurance(args.images, args.config, out_dir=args.out_dir, batch_size=args.batch_size)
        qa_result = qa.run()
        qtim_ROP.deep_rop.classify(args.image_dir, args.out_dir, self.conf_dict,
                                   skip_segmentation=args.skip_seg, batch_size=args.batch_size)

    def preprocess_images(self):

        parser = ArgumentParser()
        parser.add_argument('-c', '--config', dest='config')
        parser.add_argument('-o', '--outdir', dest='out_dir')
        parser.add_argument('-n', '--nfolds', dest='folds', default=5, type=int)
        args = parser.parse_args(sys.argv[2:])

        pipeline = qtim_ROP.preprocessing.preprocess_cross_val.Pipeline(args.config, n_folds=args.folds, out_dir=args.out_dir)
        pipeline.run()


def initialize(plus_model=None, vessel_model=None, od_model=None, quality_model=None, gpu=None):

    # Setup appdirs
    dirs = AppDirs("DeepROP", "QTIM", version=__version__)
    conf_dir = dirs.user_config_dir
    conf_file = join(conf_dir, 'config.yaml')

    if not isdir(conf_dir):
        makedirs(conf_dir)

    if not isfile(conf_file):
        config_dict = {'plus_directory': plus_model,
                       'vessel_directory': vessel_model,
                       'optic_directory': od_model,
                       'quality_directory': quality_model,
                       'gpu': str(gpu)}

        with open(conf_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    return yaml.load(open(conf_file, 'r')), conf_file


def main():
    DeepROPCommands()

