
import json
import math
from random import random

import sklearn.cluster
from numpy import ndarray

from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition
import glob


class resultObj:
    def __init__(self, filepath, box):
        self.iname = os.path.basename(filepath)
        self.bbox = np.array(box).astype(int)
class imageObj(resultObj):
    def __init__(self, resultFromBefore,image):
        self.img = image
        self.iname = resultFromBefore[0]['iname']
        self.box = []
        self.bbox = []
        self.crop = []
        self.faceEncoding = []
        rect = image
        self.dic = {}
        for i,result in enumerate(resultFromBefore):
            box = result['bbox']
            self.bbox.append(result)
            x1 = box[0]; x2 = box[0]+box[2];
            y1 = box[1]; y2 = box[1] + box[3];
            self.box.append([x1, y1, x2, y2])
            self.anotherBox = [(y1,x2,y2,x1)]
            self.crop.append(image[y1:y2,x1:x2,:])

        self.faceEncoding = face_recognition.face_encodings(self.img,self.anotherBox)
        self.dic = {"iname": self.img, "BBOX": [int(y1), int(x1), int(y1), int(x1)], "Features": self.faceEncoding}

def detect_facesDNN(input_path):
    modelFile = "dnnModel/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "dnnModel/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    resultObjList = []
    for filename in glob.glob(input_path + '/*.jpg'):
        img = cv2.imread(filename)
        rect = img

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (256, 256)), 1.0,
                                     (256, 256), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()

        confidence = faces[0,0,:,2]
        goodFaces = faces[:,:,confidence>0.64,:]

        for i in range(goodFaces.shape[2]):
            box = goodFaces[0,0,i,3:7] * np.array([w, h, w, h])

            (x, y, x1, y1) = box.astype("int")
            rect = cv2.rectangle(rect, (x, y), (x1, y1), (0, 221, 123), 2)
            box = np.array([box[0], box[1],box[2] -  box[0], box[3]-box[1]])
            resultObjList.append(resultObj(filename, box))

    return resultObjList

def detect_faces(input_path: str) -> dict:
    result_list = []
    resultObjList = detect_facesDNN(input_path)



    # # Create the cascade from cv2's model
    # # cascadeOptions = [filename for filename in glob.glob(cv2.data.haarcascades + '*.xml')]
    # cascadeOptions = [
    #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', #most reliable
    #     cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
    #     cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml',
    #     cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
    #     cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml',
    #     cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml',
    #     cv2.data.haarcascades + 'haarcascade_profileface.xml' #doesn't like to work
    # ]
    #
    # # cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    # faceCascade = []
    # for cascade in cascadeOptions:
    #     faceCascade.append(cv2.CascadeClassifier(cascade))
    #
    # # frontalCascade = faceCascade[0];
    # # profileCascade = faceCascade[1];
    # resultList = []
    # resultObjList = []
    # for filename in glob.glob(input_path + '/*.jpg'):
    #     img = cv2.imread(filename)
    #
    #
    #     # faceCascade = cv2.CascadeClassifier(cascade)
    #     # Now import the image and convert to gray
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #
    #     faces1 = [faceCascade[i].detectMultiScale(
    #         gray,
    #         scaleFactor=1.04,
    #         minNeighbors=5,
    #         minSize=(52, 52)
    #     ) for i in range(len(faceCascade))]
    #     rect = img
    #     for face in faces1:
    #         for (x, y, width, height) in face:
    #             rect = cv2.rectangle(rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
    #             box = [int(x),int(y),int(width),int(height)]
    #             resultObjList.append(resultObj(filename,box))
    #             cv2.imshow(os.path.basename(cascade) + os.path.basename(filename), img)
    #
    #     cv2.waitKey(0)
    result_list = createFile(resultObjList)
    return result_list


def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    result_list = detect_faces(input_path)
    images = []
    faceEncodings = []
    names = []
    imageDic = {}

    for filename in glob.glob(input_path + '/*.jpg'):
        img = cv2.imread(filename)
        # images.append(img)
        name = os.path.basename(filename)
        names.append(name)

        resultEntry =  list(filter(lambda result: result['iname'] == name, result_list))

        images.append(imageObj(resultEntry,img))
        faceEncodings =faceEncodings + images[-1].faceEncoding
        imageDic[name] = images[-1]

    faceLen = len(faceEncodings)
    faceEncodings = np.reshape(np.array(faceEncodings),[faceLen,128])
    k = int(K)
    # labels = [np.array(sklearn.cluster.KMeans(n_clusters=k,n_init = 2*i+10).fit(faceEncodings).labels_) for i in range(101)]
    labels = np.array(sklearn.cluster.KMeans(n_clusters=k,n_init = 10).fit(faceEncodings).labels_)
    # centers,labels = sklearn.cluster.kmeans_plusplus(faceEncodings, n_clusters=k, random_state=0)
    # kmeans.fit(faceEncodings)
    # # labels = np.array(kmeans.cluster_centers_)
    # labels = kmeans.labels_
    # no.
    # spect = SpectralClustering(5, affinity='precomputed', n_init=100,assign_labels='discretize').fit(faceEncodings)
    # spect.fit_predict(faceEncodings)
    # labels = spect.labels_
    # no
    # birch = sklearn.cluster.Birch(branching_factor = 50, n_clusters = 5, threshold = 1.5).fit(faceEncodings)
    # pred = birch.predict(faceEncodings)
    #maybe
    # db = sklearn.cluster.DBSCAN (eps=0.6, min_samples = 15).fit(faceEncodings)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True

    # labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #
    # print(labels)

    # ac = sklearn.cluster.AgglomerativeClustering(n_clusters = 5,linkage = 'ward').fit(faceEncodings)
    # labels = np.array(ac.labels_)
    # labels = np.array(labels)

    # import statistics as st
    # np.transpose(np.array([st.mode(labelset) for labelset in np.transpose(labels)]))
    # labels = st.mode(np.array(labels))
    # labels = np.transpose(labels)
    # labels = np.transpose(np.array([st.mode(labelset) for labelset in np.transpose(labels)]))
    result_list = []
    names = np.array(names)
    imageGroups = []

    for i in range(k):
        uniqueNames = np.unique(names[labels == i])
        imageGroupb = []
        for name in uniqueNames:
            imageGroupb.append(imageDic[name].crop)
        if len(imageGroupb) > 1:
            imageGroups.append(flatten(imageGroupb))
        else:
            imageGroups.append(imageGroupb)
        imageNames = names[labels == i]
        dic = {"cluster_no": int(i), "elements": []}
        dic2 = {"cluster_no": int(i), "elements": [], "features": [], "bbox": []}
        for name in imageNames:
            dic["elements"].append(str(name))
        result_list.append(dic)

    for i,imagegroup in enumerate(imageGroups):

        if len(imagegroup) > 1:
            if len(np.shape(imagegroup[0])) == 3:
                grid = np.vstack([cv2.resize(image, (48, 48), interpolation=cv2.INTER_LINEAR_EXACT) for image in imagegroup])
            else:
                grid = np.vstack([cv2.resize(image[0],(48,48),interpolation=cv2.INTER_LINEAR_EXACT) for image in imagegroup])
        else:
            grid = cv2.resize(imagegroup[0][0],(48,48), interpolation=cv2.INTER_LINEAR_EXACT)

        cv2.imwrite('cluster'+str(i)+'.jpg',np.array(grid))

    return result_list

def flatten(aList):
    return list(np.array(aList).flat)


def createFile(resultObjects):
    result = [{"iname": obj.iname, "bbox": np.array(obj.bbox).tolist()} for obj in resultObjects]
    return result
def C():
    r = random()
    b = random()
    g = random()
    return (r,g,b)
