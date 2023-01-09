"""
Written by Robert Lott for Computer Vision CSE 573
The project consists of performing face detection and clustering
I have both haar cascade and a DNN within this, the haar is commented
out as it simply was not as accurate as the model I had
"""
import Detector
import cv2
import numpy as np
import argparse
import json
import os
import sys

detectFaces = True
clusterFaces = False
clusterInput = "./faceCluster_5"
clusterOutput = "./clusters.json"
faceInput = "./validation_folder/images"
faceOutput = "./results.json"
def save(results:dict, filename:str):
    result = []
    result = results
    with open(filename, "w") as file:
        json.dump(result, file)
def main():
    if(detectFaces):
        result_list = Detector.detect_faces(faceInput)
        save(result_list, faceOutput)
    if(clusterFaces):
        result_list = Detector.detect_faces(clusterInput)
        save(result_list, clusterOutput)