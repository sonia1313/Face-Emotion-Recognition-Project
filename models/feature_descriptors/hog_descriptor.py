from skimage.feature import hog
from skimage import data, exposure
from skimage.transform import resize
from skimage import img_as_ubyte, io, color
import matplotlib.pyplot as plt
import numpy as np
"""
Adapted from:
https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
"""
class HOG_DESCRIPTOR():

  def __init__(self):

    self.des_list = []
    self.y_list = []
    self.img_list = []



  def compute_hog_descriptors(self,X,y):
    self.X = X
    X_img = X
    #X_img = [resize(i,(128,64)) for i in X]
    for i in range(len(X_img)):
      img = img_as_ubyte(color.rgb2gray(X_img[i]))
      fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

      if fd is not None:
          self.des_list.append(fd)
          self.y_list.append(y[i])
          self.img_list.append(hog_image)

  
    self.des_array = np.array(self.des_list)
    
  def show_hog_images(self):
    fig, ax = plt.subplots(2, 4, figsize=(10, 8), sharey=True)
    for i in range(4):
      hog_img = exposure.rescale_intensity(self.img_list[i], in_range=(0, 10))
      ax[0,i].imshow(hog_img, cmap = 'gray')
      ax[0,i].set_axis_off()
      ax[1,i].imshow(self.X[i], cmap = 'gray')
      ax[1,i].set_axis_off()

    fig.tight_layout()
    plt.show()

    


