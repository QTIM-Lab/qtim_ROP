import argparse
import sys
import qtim_ROP


class DeepROPCommands(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='A set of commands for segmenting and classifying retinal images',
            usage='''deeprop <command> [<args>]

            The following commands are available:
               segment                      Perform vessel segmentation using a trained U-Net
               classify_plus                Classify plus disease in retinal images using a trained CNN
            ''')

        parser.add_argument('command', help='Command to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print 'Please specify a valid command.'
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def segment(self):

        parser = argparse.ArgumentParser(
            description='Perform vessel segmentation of one or more retinal images using a U-Net')

        parser.add_argument('-i', '--images', help='Image or folder of images', dest='images', required=True)
        parser.add_argument('-o', '--out-dir', help='Output directory', dest="out_dir", default=None)
        parser.add_argument('-u', '--unet', help='Trained U-Net directory', dest='model', required=True)
        parser.add_argument('-b', '--batch-size', help='# images to load at once', default=100)

        args = parser.parse_args(sys.argv[2:])

        unet = qtim_ROP.segmentation.segment_unet.SegmentUnet(args.model, out_dir=args.out_dir)
        unet.segment_batch(args.images, batch_size=args.batch_size)


def main():

    DeepROPCommands()
