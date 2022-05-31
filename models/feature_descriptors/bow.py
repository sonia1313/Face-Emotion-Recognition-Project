import numpy as np
#this code has been adaped from lab 07
def bag_of_visual_words(des_list, y, kmeans_model, k):
  # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []

    for des in des_list:
        hist = np.zeros(k)

        idx = kmeans_model.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
      
    idx_not_empty = [i for i, x in enumerate(hist_list) if x is not None]
    hist_list = [hist_list[i] for i in idx_not_empty]
    y = [y[i] for i in idx_not_empty]
    hist_array = np.vstack(hist_list)

    return hist_array,hist_list, y
