from __future__ import division
import numpy as np
import cv2 as cv
import glob
import math


def sift(img_path):
    img = cv.imread(img_path)
    sift = cv.xfeatures2d.SIFT_create()
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imwrite('sift_keypoints2.jpg',img)
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    return kp,des

def similarity(x,y):
    dist = np.sqrt(np.sum(np.square(x - y)))
    return dist


def representation(test_path):    
    all_labels=[]
    all_representation=[]
    #word=np.full(ksize)

    #center,label = descriptors()
    test_label=[]
    
    for test in test_path:
        #print(test)
        test_representation,test_label=compute_representation(test)
        all_representation.append(test_representation)
        all_labels.append(test_label)
        test_label=np.array(test_label)    
    np.array(all_labels)
    np.save('representation.npy',all_representation)
    return all_representation,all_labels



def compute_representation(test):
    print(test)
    r=np.load('center.npy')
    test_representation=np.full(100,0)
    kp,des=sift(test)
    test_label=[]
    for x in des:
        x_label_matrix = []
        for y in r:
            x=np.array(x)
            y=np.array(y)
            x_label_matrix.append(similarity(x, y))
        x_label=x_label_matrix.index(min(x_label_matrix))
        test_representation[x_label]=test_representation[x_label]+1
        #print(x_label)
        test_label.append(x_label)
    test_representation=np.array(test_representation)
    test_label=np.array(test_label)
    return test_representation,test_label





def nomalized_representation(test,idf):
    ksize=100
    tf=[]
    word=np.full(ksize,0)
    test_representation,test_label=compute_representation(test)
    word_size=int(len(test_label))
    for i in range(0,ksize):
        word_i=np.sum(test_label.ravel()==i)
        if word_i!=0:
            word[i]=word[i]+word_i
    for i in range(0,ksize):
        tf.append(float(word[i])/float(word_size))
        #print(tf)
    tf=np.array(tf)
    idf=np.array(idf)
    tf_idf=tf*idf
    print(tf_idf)
    print(sum(tf_idf))
    np.save('tf-idf.npy',tf_idf)



if __name__ == '__main__':

    test_path='../testdata/'
    img_paths=glob.glob(test_path+'*.jpg')
    img_paths=sorted(img_paths, key=lambda name: int(name.split('/')[-1].split('.')[-2]))
    img_paths=np.asarray(img_paths)
    idf=np.load('idf.npy')
    for test in img_paths:
        nomalized_representation(test,idf)











