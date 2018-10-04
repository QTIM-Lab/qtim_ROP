from os.path import isdir, isfile
from .utils.image import find_images


class QualityAssurance:

    def __init__(self, input_imgs):

        if isdir(input_imgs):
            image_files = find_images(input_imgs)
        elif isfile(input_imgs):
            image_files = [input_imgs]
        else:
            image_files = []
            print("Please specify a valid image file or a folder of images.")
            exit(1)

        # Load the images
        pass

    def run(self):

        # Determine if this is a retina
        pass

        # Verify that it's a posterior pole image
        pass

        # Check its quality
        pass

    def is_retina(self):

        pass

    def is_posterior(self, img_batch):

        pass

    def is_quality(self):

        pass


if __name__ == "__main__":

    pass
