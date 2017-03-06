import cv2, numpy as np
from mask_retina import create_mask


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
