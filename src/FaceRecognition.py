#the following code was adapted from lab 06 
from skimage import io, color, img_as_ubyte
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np 

def detectFace(X):
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  img_copy = X.copy()
  img_gray = img_as_ubyte(color.rgb2gray(img_copy )) 
  detectedFace = face_cascade.detectMultiScale(img_gray, 1.3, 5)
  for x,y,w,h in detectedFace:
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
    img_copy = img_copy[y:y+h, x:x+w]

  return img_copy
