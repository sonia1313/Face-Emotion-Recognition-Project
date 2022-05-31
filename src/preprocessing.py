
from skimage import img_as_ubyte, io, color
import numpy as np

def convert_to_greyscale(X):
  grey_imgs = [ img_as_ubyte(color.rgb2gray(i)) for i in X]
  return np.array(grey_imgs)


def convert_to_rgb(X):
  grey_imgs = [ img_as_ubyte(color.gray2rgb(i)) for i in X]
  return np.array(grey_imgs)
