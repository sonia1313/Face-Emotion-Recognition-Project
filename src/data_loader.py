import os
import numpy as np 
from skimage import img_as_ubyte, io, color

def import_selected_data(path):
  """Load images and labels from selected directories"""
  images = []
  list_of_img_files = os.listdir(path)
  with open (os.path.join('labels',f'list_label_{path}.txt')) as fileObj:
    labels = [ int(l.split(" ")[1]) for l in fileObj.readlines()]

  file_names = [file for file in sorted(os.listdir(path)) if file.endswith('.jpg')]
  for file in file_names:
    images.append(io.imread(os.path.join(path, file)))

  return np.array(images), np.array(labels)
    
def import_personal_data(path):
  """Load images and labels from selected directories"""
  images = []
  list_of_img_files = os.listdir(path)
  with open (os.path.join('personal-test-labels',f'list_labels_personal.txt')) as fileObj:
    labels = [ int(l.split(" ")[1]) for l in fileObj.readlines()]

  file_names = [file for file in sorted(os.listdir(path)) if file.endswith('.jpg')]
  for file in file_names:
    images.append(io.imread(os.path.join(path, file)))

  return np.array(images), np.array(labels)

def shuffleData(X, y):
    #assert len(X) == len(y)
    perms = np.arange(len(X))
    np.random.shuffle(perms)

    X = X[perms]
    y = y[perms]

    return X, y
