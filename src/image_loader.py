#show predicted images
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def display_images(X,y,predicted):
  X_test, y_test, predicted = shuffle(X , y, predicted)


  fig, axes = plt.subplots(1, 4, figsize=(14, 7), sharex=True, sharey=True)
  ax = axes.ravel()

  for i in range(4):
      ax[i].imshow(X_test[i])
      ax[i].set_title(f'Label: {y_test[i]} \n Prediction: {predicted[i]}')
      ax[i].set_axis_off()
  fig.tight_layout()
  plt.show()
