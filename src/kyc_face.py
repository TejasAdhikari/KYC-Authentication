import tensorflow as tf
import keras
import model_images
from keras import backend as K
from keras import applications
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import model_from_json, Model, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import model_from_json
from PIL import Image
import os
import numpy as np
import cv2
import time
import pickle


K.set_learning_phase(0)
model_name = 'vgg_face'

def vgg_face(weights_path = 'D:\T\SpitHackathon\src/vgg_face_weights.h5'):

    model = keras.Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def vgg_16():
    model = keras.applications.vgg16.VGG16()
    return model

def faceNet():
    model = load_model('facenet_keras.h5')
    return model

def get_model(name = model_name):
    if name == 'vgg_16': return vgg_16()
    elif name == 'vgg_face': return vgg_face()
    elif name == 'faceNet': return faceNet()
    else: return None

model = get_model(model_name)
face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
#face_descriptor = K.function([model.layers[0].input], [model.layers[-2].output])
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_names = [layer.name for layer in model.layers]
epsilon_cos = 0.65 #cosine similarity (needs to be updated)
epsilon_dist = 69 #euclidean distance (needs to be updated)

def preprocess_image(img, model_name = model_name, matrix = False):
    if model_name == 'faceNet':
        if matrix:
            mat  = []
            for im in img:
                image = im.copy()
                image = image.resize((160, 160))
                image = img_to_array(image)
                mean, std = image.mean(), image.std()
                image = (image - mean) / std
                mat.append(image)
            return np.asarray(mat)
        else:
            img = img.resize((160, 160))
            img = img_to_array(img)
            mean, std = img.mean(), img.std()
            img = (img - mean) / std
    else:
        img_ = np.asarray(img)
        img_ = img.resize((224, 224))
        img = img_to_array(img_)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def getFeatureVector(img, matrix = False, preprocess = True, pre_compute = False):
    img_representation = None
    if matrix:
        if preprocess:
            if pre_compute: img_representation = face_descriptor([preprocess_image(img, matrix = True)])
            else: img_representation = face_descriptor.predict(preprocess_image(img, matrix = True))
        else:
            if pre_compute: img_representation = face_descriptor([img])
            else: img_representation = face_descriptor.predict(img)
    else:
        if preprocess:
            if pre_compute: img_representation = face_descriptor([preprocess_image(img)])
            else: img_representation = face_descriptor.predict(preprocess_image(img))[0,:]
        else:
            if pre_compute: img_representation = face_descriptor([img])
            else: img_representation = face_descriptor.predict(img)[0,:]
    return img_representation

def getFeatureVectorBeta(img, matrix = False, preprocess = True, pre_compute = True):
    img_representation = None

    if preprocess: img_representation = face_descriptor([preprocess_image(img, matrix = matrix)])
    else: img_representation = face_descriptor([img])

    return img_representation

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(source_representation, np.transpose(test_representation))
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return (a / (np.sqrt(b) * np.sqrt(c)))

def findCosineDifference(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findL1Norm(source_representation, test_representation):
    l1_distance = source_representation - test_representation
    l1_distance = np.absolute(l1_distance)
    l1_distance = np.sum(l1_distance)
    return l1_distance

def verifyScore(euclidean_distance, cosine_similarity):
    if((euclidean_distance < epsilon_dist) & (cosine_similarity > epsilon_cos)): return True
    else: return False


def verifyFace(img1, img2, print_score = False):
    img1_representation = getFeatureVector(img1)
    img2_representation = getFeatureVector(img2)

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    l1_distance = findL1Norm(img1_representation, img2_representation)

    if print_score: print('Cos: ' + str(cosine_similarity) + ' Dist: ' + str(euclidean_distance) + ' L1 Dist: ' + str(l1_distance))
    #cumulative_diff = cosineDifference*(euclidean_distance)*(l1_distance)

    #print('cumulative_diff: ', cumulative_diff)

    return verifyScore(euclidean_distance, cosine_similarity)

def verifyFaceVector(img1, img2_representation, print_score = False):
    img1_representation = getFeatureVector(img)

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    l1_distance = findL1Norm(img1_representation, img2_representation)

    if print_score: print('Cos: ' + str(cosine_similarity) + ' Dist: ' + str(euclidean_distance) + ' L1 Dist: ' + str(l1_distance))

    return verifyScore(euclidean_distance, cosine_similarity)

def verifyVecs(img1_representation, img2_representation, print_score = False):

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    l1_distance = findL1Norm(img1_representation, img2_representation)

    if print_score: print('Cos: ' + str(cosine_similarity) + ' Dist: ' + str(euclidean_distance) + ' L1 Dist: ' + str(l1_distance))

    return verifyScore(euclidean_distance, cosine_similarity)

def verifyVecMat(vector, matrix, print_score = False, first_match = False, best_match = True):

    a = np.matmul(matrix, np.transpose(vector))
    b = np.sum(vector*vector)
    c = np.sum(matrix*matrix, axis = -1)
    cosine_similarity =  a / (np.sqrt(b) * np.sqrt(c))

    euclidean_distance = np.asarray(matrix) - np.asarray(vector)
    euclidean_distance = np.sum(euclidean_distance*euclidean_distance, axis = -1)
    euclidean_distance = np.sqrt(euclidean_distance)

    if print_score: print('Cos: '+str(cosine_similarity)+' Dist: '+str(euclidean_distance))
    if best_match:
        e_by_c = cosine_similarity/(euclidean_distance*euclidean_distance)
        max_id = np.argmax(e_by_c)
        if((euclidean_distance[max_id]<epsilon_dist)&(cosine_similarity[max_id]>epsilon_cos)): return max_id
        else: return None
    euclidean_distance =  [eu_dist < epsilon_dist for eu_dist in euclidean_distance]
    cosine_similarity =  [cos_sim > epsilon_cos for cos_sim in cosine_similarity]
    match_vec = [euclidean_distance[i] & cosine_similarity[i] for i in range(len(cosine_similarity))]
    if first_match:
        try: return match_vec.index(True)
        except ValueError: return None
    return match_vec

def test_get_feature(img):
    tic = time.time()
    img_representation = face_descriptor([preprocess_image(img), 0])
    print('Time Required: ', time.time() - tic)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

def verify_images(img_1, img_2):
	vec1 = getFeatureVector(img_1, matrix = False, preprocess = True, pre_compute = False)
	vec2 = getFeatureVector(img_2, matrix = False, preprocess = True, pre_compute = False)
	return verifyVecs(np.asarray(vec1), np.asarray(vec2), print_score = True)
