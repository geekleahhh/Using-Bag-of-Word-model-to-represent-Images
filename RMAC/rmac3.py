from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K

from keras.applications import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

from sklearn.decomposition import PCA 
from sklearn.preprocessing import Normalizer

import scipy.io
import numpy as np
import glob


def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):
    print('input_shape: %s' %str(input_shape))

    # Load VGG16
    vgg16_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)

    # Freeze the layers
    for layer in vgg16_model.layers:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg16_model.layers:
        print(layer, layer.trainable)


    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    # every matrix use roi pooling, get? (1,14) vector, sum them, get a (1, 14) vector
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-5].output, in_roi])

    # Normalization
    #every vecter(1, 14) normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)    
    model=Model([vgg16_model.input,in_roi], x)
    return model



def preprocess(file, IMG_SIZE):
    img = image.load_img(file)
    scale = IMG_SIZE
    new_size = (int(np.ceil(scale)), int(np.ceil(scale)))
    print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def ComputeRMAC(x):
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1]) #Wmap, Hmap means extract from a root 6 times
    regions = rmac_regions(Wmap, Hmap, 10)
    # Load RMAC model
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
    # Compute RMAC vector
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    print(RMAC)
    RMAC = np.reshape(RMAC,(-1,512))
    print(RMAC.shape)
    return RMAC


def PCARMAC(RMAC):
    #pca
    pca=PCA(n_components=10, whiten = True)
    newRMAC=pca.fit_transform(RMAC)

    #addition
    RMAC=np.reshape(np.sum(newRMAC, axis=0),(-1,10))

    #L2 normalization
    norm2 = Normalizer(copy=True,norm='l2')
    transformer = norm2.fit(RMAC)
    print('RMAC size: %s' % str(RMAC.shape))
    RMAC=transformer.transform(RMAC.astype(float))
    return(RMAC)





    


if __name__ == "__main__":
    # Load sample image
    DATA_DIR= '/Users/yaoweili/Documents/keras_rmac-master/data/'
    img_paths=glob.glob(DATA_DIR+'*.jpg')
    IMG_SIZE=244
    for file in img_paths:
        #x = preprocess(file, IMG_SIZE)
        #RMAC = ComputeRMAC(x)
        #np.save('RMAC.npy',RMAC)
        RMAC=np.load('RMAC.npy')
        PCA_RMAC = PCARMAC(RMAC)
        print(PCA_RMAC)
        
        