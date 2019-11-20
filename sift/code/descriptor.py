from __future__ import division
import numpy as np
import cv2 as cv
import glob
import math
# from libKMCUDA import kmeans_cuda


def sift(img_path):
    img = cv.imread(img_path)
    sift = cv.xfeatures2d.SIFT_create()
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    return kp,des





def descriptors():
    ksize=100
    des = None
    img_descs = None
    center = None
    word = np.full(ksize,0)

    idf = np.full(ksize,0)
    np.array(img_descs)
    image_path='../data/'
    img_paths=glob.glob(image_path+'*.jpg')
    #img_paths=np.asarray(img_paths)
    img_paths=sorted(img_paths, key=lambda name: int(name.split('/')[-1].split('.')[-2]))


    step = int(len(img_paths) / 200)
    for i in range(0, 200):
        print(i)
        img_descs = center
        for image in img_paths[i*step: (i+1)*step]:

            trans_image_path=image_path+image.split('/')[-1].split('.')[-2]+'/'
            trans_img_path=glob.glob(trans_image_path+'*.jpg')
            img = cv.imread(image)
            h, w, channels = img.shape
            kp, des=sift(image)

            if img_descs is not None:
                img_descs = np.vstack((img_descs, des))
            else:
                img_descs = des
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv.kmeans(img_descs,ksize,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    np.save('center.npy',center)
    return center, label



def similarity(x,y):
    dist = np.sqrt(np.sum(np.square(x - y)))
    return dist

def database(img_paths):
    final_index=[]
    word=np.full(100,0)
    label_index=np.empty(shape=[0, 2])
    all_representation,all_labels=representation(img_paths)
    all_labels=np.array(all_labels)
    print(all_labels)
    ksize=100
    #final_index
    word2=np.full(ksize,0)
    j=0
    for labels in all_labels:
        j=j+1
        for i in range(0,ksize):
            #print word2[i]
            word2[i]=np.sum(labels.ravel()==i)
            x=any(labels.ravel()==i)
            #print(word2[i])
            #print(any(label.ravel()==i))
            if x==False:
                print("attention!",j,i)
            else:
                    print(j,i)
                    word[i]=word[i]+1
                    label_index=np.append(label_index,[[j,i]],axis=0)
                    print(label_index)
                #print(word[i],i)
        #print(sum(word2))
    idf = []
    idff=np.full(ksize,0)
    for i in range(0,ksize):
        #idff[i]=(float(len(img_paths))/float(word[i]))
        a = int(len(img_paths))
        b = int(float(word[i]))+1
        idf.append(round(math.log(float(a)/float(b)),2))
        final_index.append(label_index[:,0][label_index[:,1]==i])
        print(label_index[:,1]==i)
        print(label_index[:,0][label_index[:,1]==i])
        #idf[i]=math.log(idff[i])
    final_index=np.array(final_index)
    print(word)
    print(idf)
    print(final_index)
    return 0


def representation(test_path):
    all_labels=[]
    all_representation=[]
    #word=np.full(ksize)

    center,label = descriptors()
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




def search(test_path,img_paths):
    findmax=[]
    r,v=compute_representation(test_path)
    all_representation=np.load('representation.npy')
    for i in all_representation:
        findmax.append(similarity(i,r))
    maxone_index=findmax.index(min(findmax))
    #print(maxone_index)
    maxone=img_paths[maxone_index]
    print(maxone)


def nomalized_representation(test):
    ksize=100
    tf=[]
    word=np.full(ksize,0)
    word_size=int(len(test_label))
    test_representation,test_label=compute_representation(test)
    for i in range(0,ksize):
        word_i=np.sum(labels.ravel()==i)
        if word_i!=0:
            word[i]=word[i]+word_i
        for i in range(0,ksize):
            tf.append(float(word[i])/float(word_size))
            print(tf)






if __name__ == '__main__':
    descriptors()
