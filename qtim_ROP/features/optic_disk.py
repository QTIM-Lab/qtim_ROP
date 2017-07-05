import matplotlib.pyplot as plt
from ..segmentation.mask_retina import *


def hough_vessels(im_arr):

    # Perform edge detection and mask
    mask = create_mask(im_arr, erode=50)
    edges = cv2.Canny(im_arr, 50, 150, apertureSize=3)
    masked_edges = edges * mask

    fig1, ax1 = plt.subplots()
    plt.imshow(masked_edges)
    plt.show()

    # Draw lines
    lines = cv2.HoughLines(masked_edges, 200, np.pi, 20)
    plt.imshow(draw_lines(lines, rgb2gray(im_arr)))
    plt.show()


def draw_lines(lines, img):

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img

if __name__ == '__main__':


    import sys

    im_arr = cv2.imread(sys.argv[1])[:,:,::-1]
    hough_vessels(im_arr)
