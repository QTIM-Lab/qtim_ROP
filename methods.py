import cv2, numpy as np
from mask_retina import create_mask
from skimage.morphology import binary_erosion, selem

# https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/
# kaggleDiabeticRetinopathy/preprocessImages.py
def kaggle_BG(a, scale):

    # a = scale_radius(a, scale)
    # b = numpy.zeros(a.shape)
    # cv2.circle(b, (a.shape[1] / 2, a.shape[0] / 2), int(scale * cf), (1, 1, 1), -1, 8, 0)
    b = create_mask(a, erode=None)
    b = np.stack([b]*3, axis=2)
    aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
    return aa.astype(np.uint8)


# https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/
# kaggleDiabeticRetinopathy/preprocessImages.py
def scale_radius(img, scale):
    x = img[img.shape[0] / 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def normalize_channels(img):
    for colorband in xrange(img.shape[2]):
        img[:, :, colorband] = image_histogram_equalization(img[:, :, colorband])
    return img

# http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
def image_histogram_equalization(image, number_bins=256):

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def binary_morph(img, min_size=None):

    if min_size is None:  # default to 10% of largest image dimension
        min_size = float(max(img.shape)) * .1

    if len(img.shape) == 3:  # flatten if RGB image
        img = np.mean(img, 2).astype(np.uint8)

    # Apply binary threshold and erode
    ret, thresh_im = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Connected component labelling
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_im)
    mask = np.zeros_like(labels)

    # Loop through areas in order of size
    areas = [s[4] for s in stats]
    sorted_idx = np.argsort(areas)

    for lidx, cc in zip(sorted_idx, [areas[s] for s in sorted_idx][:-1]):

        if cc > min_size:
            mask[labels == lidx] = 1

    return np.dstack([img * mask] * 3).astype(np.uint8)


if __name__ == '__main__':

    import sys
    from os.path import dirname, join

    img = cv2.imread(sys.argv[1])

    for ms in range(150, 250, 10):

        masked_img = binary_morph(img, ms)
        cv2.imwrite(join(dirname(sys.argv[1]), 'seg_m{}.bmp'.format(ms)), masked_img)
