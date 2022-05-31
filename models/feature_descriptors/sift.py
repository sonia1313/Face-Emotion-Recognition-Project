"""The following code has been adapted from LAB 07"""
import cv2
from skimage import img_as_ubyte, io, color
import matplotlib.pyplot as plt
import numpy as np

class SIFT_DETECTOR():

  def __init__(self):
    
    self.des_list = []
    self.y_list = []
    self.kp_list = []
    self.img_list = []


  def identify_keypoints(self, X_train_img,y):
    #fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

    #X_train_img = X.reshape(-1, 100, 100)
    sift = cv2.SIFT_create()
    for i in range(0,len(X_train_img)):
      #print(i)
      img = img_as_ubyte(color.rgb2gray(X_train_img[i]))
      kp, des = sift.detectAndCompute(img, None)

      if des is not None:
        self.des_list.append(des)
        self.kp_list.append(kp)
        self.img_list.append(img)
        self.y_list.append(y[i])
          

    self.des_array = np.vstack(self.des_list)
  

  def show_SIFT_keypoints(self):
    fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)
    for i in range(4):
      img_with_SIFT = cv2.drawKeypoints(self.img_list[i], self.kp_list[i], self.img_list[i])
      ax[i].imshow(img_with_SIFT)
      ax[i].set_axis_off()
    fig.tight_layout()
    plt.show()

    


def get_feature_descriptor(X_test, y_test, kmeans,k):
  hist_list = []
  sift = cv2.SIFT_create()
  for i in range(len(X_test)):
      img = img_as_ubyte(color.rgb2gray(X_test[i]))
      kp, des = sift.detectAndCompute(img, None)

      if des is not None:
          hist = np.zeros(k)

          idx = kmeans.predict(des)

          for j in idx:
              hist[j] = hist[j] + (1 / len(des))

          # hist = scale.transform(hist.reshape(1, -1))
          hist_list.append(hist)

      else:
          hist_list.append(None)

  # Remove potential cases of images with no descriptors
  idx_not_empty = [i for i, x in enumerate(hist_list) if x is not None]
  hist_list = [hist_list[i] for i in idx_not_empty]
  y_test = [y_test[i] for i in idx_not_empty]
  hist_array = np.vstack(hist_list)
  return hist_array, hist_list, y_test



      




