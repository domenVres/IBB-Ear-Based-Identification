import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch.nn as nn
from PIL import Image


class Preprocess:

    def histogram_equalization_rgb(self, img, BGR=True):
        # Simple preprocessing using histogram equalization
        # https://en.wikipedia.org/wiki/Histogram_equalization

        if BGR:
            intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            intensity_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        if BGR:
            img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)
        else:
            img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2RGB)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    def edge_enhancement(self, img):
        # Edge enhancement using edge detection filter
        # Filter source: https://en.wikipedia.org/wiki/Kernel_(image_processing)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    def image_sharpening(self, img, kernel_size=(5, 5), sigma=1.0):
        # Sharpen the image by subtracting the Gaussian blur filter from it
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharp = 2*img - blurred

        # Handle the values below 0
        sharp = np.maximum(sharp, np.zeros(sharp.shape))
        # Handle the values above 255
        sharp = np.minimum(sharp, 255*np.ones(sharp.shape))
        # Round floats
        sharp = np.round(sharp).astype("uint8")

        return sharp


"""
Same preprocessing techniques represented as pytorch transforms so they can be used by CNNs
"""
class HistogramEqualization(nn.Module):
    def forward(self, image):
        p = Preprocess()
        image = np.array(image)
        image = p.histogram_equalization_rgb(image, BGR=False)
        image = Image.fromarray(image)

        return image


class EdgeEnhancement(nn.Module):
    def forward(self, image):
        p = Preprocess()
        image = np.array(image)
        image = p.edge_enhancement(image)
        image = Image.fromarray(image)

        return image


class ImageSharpening(nn.Module):
    def forward(self, image):
        p = Preprocess()
        image = np.array(image)
        image = p.image_sharpening(image)
        image = Image.fromarray(image)

        return image


if __name__=="__main__":
    fname = sys.argv[1]
    img = cv2.imread(f"../data/ears/test/{fname}")

    f, axarr = plt.subplots(2, 2, figsize=(10, 10))
    axarr[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axarr[0, 0].set_title("Original")

    preprocess = Preprocess()
    img_hist = preprocess.histogram_equalization_rgb(img)
    axarr[0, 1].imshow(cv2.cvtColor(img_hist, cv2.COLOR_BGR2RGB))
    axarr[0, 1].set_title("Histogram equalization")

    img_edge = preprocess.edge_enhancement(img)
    axarr[1, 0].imshow(cv2.cvtColor(img_edge, cv2.COLOR_BGR2RGB))
    axarr[1, 0].set_title("Edge enhancement")

    img_sharp = preprocess.image_sharpening(img)
    axarr[1, 1].imshow(cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB))
    axarr[1, 1].set_title("Image sharpening")

    plt.show()