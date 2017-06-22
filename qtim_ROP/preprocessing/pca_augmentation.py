import h5py
import pandas as pd
from os.path import splitext
import numpy as np
from sklearn.decomposition import PCA


def pca_augmentation(data_h5, excel_path):

    f = h5py.File(data_h5, 'r')

    df1 = pd.read_excel(excel_path, sheetname=0, header=1)
    df1 = df1.rename(columns=lambda x: x.strip()).set_index('Image')  # strip whitespace

    df2 = pd.read_excel(excel_path, sheetname=1, header=1)
    df2 = df2.rename(columns=lambda x: x.strip()).set_index('Image')  # strip whitespace
    df = pd.concat([df1, df2])

    X = preprocess_data(f)
    X_mean = np.mean(X, axis=0)
    X = X - X_mean

    # PCA
    pca = PCA().fit(X)


def preprocess_data(f):

    X = []

    for img, name in zip(f['data'], f['filenames']):

        name = splitext(name)[0]

        if name.find('od'):
            eye = 'od'
        elif name.find('os'):
            eye = 'os'
        else:
            continue

        if eye == 'od':
            img = np.fliplr(img)

        # Normalize image
        img = img.astype(np.float16)
        img_min, img_max = np.min(img), np.max(img)
        img_norm = (img - img_min) / (img_max - img_min)
        X.append(img_norm)

    X = np.asarray(X)
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

if __name__ == '__main__':

    import sys
    pca_augmentation(sys.argv[1], sys.argv[2])