from os.path import *
from os import makedirs, listdir
from glob import glob
import numpy as np
import keras.backend as K
from keras.preprocessing import image
from keras.models import load_model, model_from_json
import pandas as pd
from PIL import Image
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from .utils.common import find_images_recursive
from .utils.image import find_images
from .retinaunet.lib.pre_processing import my_PreProc
from .segmentation.optic_disc import od_statistics


class QualityAssurance:

    def __init__(self, input_imgs, config, out_dir, batch_size=32, recursive=True, debug=False):

        self.config = config
        self.out_dir = out_dir
        self.od_dir = join(self.out_dir, 'optic_disk')

        if not exists(self.od_dir):
            print("Creating optic disk directory")
            makedirs(self.od_dir)
        else:
            print(f'Optic disk folder already exists: {self.od_dir}')

        self.batch_size = batch_size

        print("Searching for files")
        if isdir(input_imgs):
            search_func = find_images_recursive if recursive else find_images
            image_files = search_func(input_imgs)
        elif isfile(input_imgs):
            image_files = [input_imgs]
        else:
            image_files = None
            print("Please specify a valid image file or a folder of images.")
            exit(1)

        self.image_files = image_files
        print(f'Found {len(self.image_files)} files, and {len(listdir(self.od_dir))} already appear to be segmented')
        self.quality_path = config['quality_directory']
        # self.retina_path = config['retina_directory']
        self.posterior_path = config['optic_directory']

        self.csv_out = join(self.out_dir, 'QA.csv')
        if isfile(self.csv_out):
            self.results = pd.read_csv(self.csv_out, index_col=0)
        else:
            index = [splitext(basename(f))[0] for f in self.image_files]
            self.results = pd.DataFrame([], index=index)
            self.results['Full path'] = self.image_files

    def run(self):

        # Determine if this is a retina
        pass  # TODO need from Aaron

        # Check its quality
        print("Estimating image quality...")
        self.is_quality()

        # Verify that it's a posterior pole image
        print("Verifying images are posterior pole...")
        self.is_posterior()

        print(self.results)
        self.results.to_csv(self.csv_out)
        return self.results

    def is_retina(self):

        pass

    def is_quality(self):

        # Load the model
        quality_model = load_model(glob(join(self.quality_path, '*.h5'))[0])
        results = []

        for (file_names, batch), _ in self.batch_loader():

            index = [splitext(basename(f))[0] for f in file_names]
            batch = batch / 255
            predictions = np.squeeze(quality_model.predict(batch))
            series = pd.Series(predictions, index=index)
            results.append(series)

        concat_series = pd.concat(results, axis=0)
        concat_series = concat_series[~concat_series.index.duplicated(keep='first')]  # ridiculous hack to deal with duplicate indices

        self.results['Quality'] = concat_series
        self.results.to_csv(self.csv_out)

    def is_posterior(self, tol_pixels=1000):

        # Load the model
        od_json = open(glob(join(self.posterior_path, '*.json'))[0], 'r').read()
        od_weights = glob(join(self.posterior_path, '*.h5'))[0]
        od_model = model_from_json(od_json)  # note: only works in Keras 2.1.6 / tensorflow-gpu 1.8
        od_model.load_weights(od_weights)

        results = []
        centroid = (240, 240)

        for (file_names, batch), is_raw in self.batch_loader(has_output=True, out_dir=self.od_dir, target_size=(480, 640)):

            # Run inference, if the data loaded is the raw data
            if is_raw:
                prep_batch = my_PreProc(batch)
                prep_batch = prep_batch[:, :, 80:-80, :].transpose((0, 3, 1, 2))  # crop to 480 x 480 square
                predictions = od_model.predict(prep_batch)
                self.save_batch(predictions, file_names, self.od_dir)
            else:
                predictions = batch.transpose((0, 3, 1, 2)).astype(np.float32) / 255.  # just load the previous predictions

            # Calculate statistics
            index = [splitext(basename(f))[0] for f in file_names]
            batch_stats = pd.DataFrame(
                [od_statistics(img[0], filename) for img, filename in zip(predictions, index)]).set_index('filename')

            batch_stats['euc_distance_centroid'] = batch_stats.apply(lambda p: euclidean(centroid, p[['x', 'y']]) if p['no_objects'] > 0 else None, axis=1)
            batch_stats['is_posterior'] = (batch_stats['no_objects'].astype(int) == 1) & (batch_stats['euc_distance_centroid'] < tol_pixels)

            results.append(batch_stats)
            print(len(results))

        # Compile final DataFrame
        result_df = pd.concat(results, axis=0)
        self.results = self.results.join(result_df)
        # result_df['euc_distance_centroid'] = result_df.apply(lambda p: euclidean(centroid, p[['x', 'y']]) if p['no_objects'] > 0 else None, axis=1)
        # self.results['euc_distance_centroid'] = result_df['euc_distance_centroid']
        # self.results['is_posterior'] = (result_df['no_objects'].astype(int) == 1) & (result_df['euc_distance_centroid'] < tol_pixels)
        # self.results['no_objects'] = result_df['no_objects']
        # self.results['x'] = result_df['x']
        # self.results['y'] = result_df['y']
        print(self.results)

    def save_batch(self, pred, file_names, out_dir):

        for img, out_file in zip(pred, file_names):
            Image.fromarray((255. * img[0]).astype('uint8')).save(join(out_dir, basename(out_file)))

    def batch_loader(self, has_output=False, out_dir=None, target_size=(150, 150)):

        skipped = []

        def load_batch(batch):

            batch_filenames, batch_images = [], []
            for filename in batch:

                try:
                    resized = np.asarray(image.load_img(filename, target_size=target_size))
                    batch_filenames.append(filename)
                    batch_images.append(resized)
                except OSError:
                    print(f'{filename} was skipped')
                    skipped.append(filename)
                    pd.Series(skipped).to_csv('skipped.csv')
                    continue

            batch_tensor = np.stack(batch_images, axis=0)
            return batch_filenames, batch_tensor

        for start in tqdm(range(0, len(self.image_files), self.batch_size)):

            end = min(start + self.batch_size, len(self.image_files))
            batch = self.image_files[start:end]

            if has_output and out_dir:
                analyzed_batch = [join(out_dir, basename(f)) for f in batch]
                if not all([isfile(f) for f in analyzed_batch]):
                    yield load_batch(batch), True
                else:
                    yield load_batch(analyzed_batch), False
            else:
                yield load_batch(batch), True


if __name__ == "__main__":

    pass
