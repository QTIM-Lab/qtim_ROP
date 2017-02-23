import cv2, numpy
from scipy.misc import imresize


def resize(f, *args):

    width, height = int(args[0]), int(args[1])
    im = cv2.imread(f)

    im_shrunk = imresize(im, (width, height), interp='bicubic')
    return im_shrunk

# https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/
# kaggleDiabeticRetinopathy/preprocessImages.py
def kaggle_BG(f, *args):

    scale, cf = args[0], args[1]
    a = cv2.imread(f)
    a = scale_radius(a, scale)
    b = numpy.zeros(a.shape)
    cv2.circle(b, (a.shape[1] / 2, a.shape[0] / 2), int(scale * cf), (1, 1, 1), -1, 8, 0)
    aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
    return aa


# https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/
# kaggleDiabeticRetinopathy/preprocessImages.py
def scale_radius(img, scale):
    x = img[img.shape[0] / 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


