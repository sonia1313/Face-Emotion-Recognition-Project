from data_loader import import_selected_data, shuffleData, import_personal_data
from image_loader import display_images
from joblib import load
from FaceRecognition import detectFace
import os
from hog_descriptor import HOG_DESCRIPTOR
from sift import get_feature_descriptor
from sklearn.utils import shuffle
from sklearn import metrics
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np 

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Colab Notebooks/Face-Emotion-Recognition-Project' 
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)

def EmotionRecognition(path_to_testset, model):
  X, y = getData(path_to_testset)
  #print(len(y))
  if model == 'HOG_MLP':
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/trained_hog_mlp.joblib')
    trained_hog_mlp = load(filename)
    X_test, y_test = X.copy(), y.copy()

    hog_obj = HOG_DESCRIPTOR()
    hog_obj.compute_hog_descriptors(X_test,y_test)
    test_hog_features = hog_obj.des_array
    predicted = trained_hog_mlp.predict(test_hog_features).tolist()

    display_images(X_test,y_test,predicted)
  elif model == 'SIFT_MLP':
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/trained_sift_mlp.joblib')
    trained_sift_mlp = load(filename)
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/trained_hog_km_means.joblib')
    kmeans_model = load(filename)
    X_test, y_test = X.copy(), y.copy()

    test_hist_array, test_hist_list, y_test = get_feature_descriptor(X_test, y_test, kmeans_model, 100)
    predicted_svm = trained_sift_mlp.predict(test_hist_array).tolist()

    idx_not_empty = [i for i, x in enumerate(test_hist_list) if x is not None]
    X_test = [X_test[i] for i in idx_not_empty]

    display_images(X_test, y_test, predicted_svm)
  elif model == 'SIFT_BOW_SVM':
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/trained_sift_bow_svm.joblib')
    trained_sift_svm = load(filename)
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/baseline_kmm.joblib')
    kmeans_model = load(filename)
    X_test, y_test = X.copy(), y.copy()

    test_hist_array, test_hist_list, y_test = get_feature_descriptor(X_test, y_test, kmeans_model, 100)
    predicted_svm = trained_sift_svm.predict(test_hist_array).tolist()

    idx_not_empty = [i for i, x in enumerate(test_hist_list) if x is not None]
    X_test = [X_test[i] for i in idx_not_empty]

    display_images(X_test, y_test, predicted_svm)
  elif model == 'HOG_SVM':
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/trained_hog_svm.joblib')
    trained_hog_svm = load(filename)
    X_test, y_test = X.copy(), y.copy()

    hog_obj = HOG_DESCRIPTOR()
    hog_obj.compute_hog_descriptors(X_test,y_test)
    test_hog_features = hog_obj.des_array
    predicted = trained_hog_svm.predict(test_hog_features).tolist()

    idx_not_empty = [i for i, x in enumerate(predicted) if x is not None]
    y_test = [y_test[i] for i in idx_not_empty]

    display_images(X_test,y_test,predicted)
  elif model == 'VGG16_MLP' and path_to_testset == 'test':
    X_test, y_test = X.copy(), y.copy()
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/extracted_features.npy')
    extracted_test_features = np.load(filename)
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/cnn_mlp_.joblib')
    vgg_16_mlp = load(filename)

    predicted = vgg_16_mlp.predict(extracted_test_features).tolist()
    
    idx_not_empty = [i for i, x in enumerate(predicted) if x is not None]
    y_test = [y_test[i] for i in idx_not_empty]
    
    display_images(X_test,y_test,predicted)
  elif model == 'VGG16_MLP' and path_to_testset == 'personal-test-images':
    X_test, y_test = X.copy(), y.copy()
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/extracted_personal_test_features.npy')
    extracted_personal_test_features = np.load(filename)
    filename = os.path.join(GOOGLE_DRIVE_PATH, 'models/lib/cnn_mlp_.joblib')
    vgg_16_mlp = load(filename)

    predicted = vgg_16_mlp.predict(extracted_personal_test_features).tolist()
    
    idx_not_empty = [i for i, x in enumerate(predicted) if x is not None]
    y_test = [y_test[i] for i in idx_not_empty]
    
    display_images(X_test,y_test,predicted)


def getData(path_to_testset):

  if path_to_testset == 'personal-test-images':
    X,y = import_personal_data(path_to_testset)

    cropped_X = []
    for i in range(len(X)):
      f = detectFace(X[i])
      cropped_X.append(resize(f,(100,100)))

    personalX, personalY = np.array(cropped_X), y

    return personalX, personalY
  else:
    X,y = import_selected_data(path_to_testset)

    return X,y







    


    
  
    