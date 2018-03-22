from . import qtim_ROP

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-s', '--splits', dest='splits', help='Directory of trained CNNs', required=True)
    parser.add_argument('-i', '--raw-images', dest='raw_images', help='Directory of all raw images', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory for results', required=True)
    parser.add_argument('-r', '--rf', dest='use_rf', help='Use random forest?', action='store_true', default=False)

    args = parser.parse_args()
    qtim_ROP.evaluation.cross_validation.run_cross_val(args.splits, args.raw_images, args.out_dir, use_rf=args.use_rf)
