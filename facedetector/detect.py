import sys
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import dlib

#####################################################################################

#FACE DETECTION CLASS
class FADetector():
    def __init__(self):
        self.fa2 = dlib.get_frontal_face_detector()
        self.fa3 = cv2.dnn.readNetFromCaffe("./model/deploy.prototxt.txt","./model/res10_300x300_ssd_iter_140000.caffemodel")
        self.fa4 = dlib.cnn_face_detection_model_v1("./model/mmod_human_face_detector.dat")
        self.thresh = 0.5

    #DLIB SIMPLE LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def cv2dnn_facedetection(self,rgb,pad=20):
        h,w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb,(300,300)),1.0,(300,300),(103.93,116.77,123.68))
        self.fa3.setInput(blob)
        detections = self.fa3.forward()

        #get driver bounding box based on rightmost position
        rightmost = -1
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            if confidence > 0.7 and box[0] > rightmost:
                rightmost = box[0]
                box = box.astype("int")
                bbox = dlib.rectangle(box[0],box[1],box[2],box[3])
        if rightmost == -1: return

        # return output
        print(f"top: {bbox.top()}")
        print(f"bot: {bbox.bottom()}")
        print(f"left: {bbox.left()}")
        print(f"right: {bbox.right()}")
        bounded_img = cv2.rectangle(rgb,(bbox.left(),bbox.top()), (bbox.right(),bbox.bottom()),[255,0,0],1)
        cv2.imshow('Detection',cv2.cvtColor(bounded_img, cv2.COLOR_RGB2BGR))

        return rgb[bbox.top():bbox.bottom(),bbox.left():bbox.right()]

    #CNN FACE DETECTION FROM DLIB
    def dlibcnn_facedetection(self,rgb,save=False):
        dets = self.fa4(rgb,0)
        d = dets[0]
        return rgb[d.rect.top():d.rect.bottom(),d.rect.left():d.rect.right()]

######################################################################################
######################################################################################
######################################################################################
#EVALUATION FUNCTIONS

#OPENCV'S FACE DETECTION MODULE FROM ITS DNN LIBRARY
def test_getface(fin, mode=''):
    fa_detector = FADetector()

    rgb = plt.imread(fin)

    #PERFORM FACE DETECTION
    st_time = time.time()
    if mode == 'cv2dnn':
        img = fa_detector.cv2dnn_facedetection(rgb)
    elif mode == 'yolo-cpu':
        img = fa_detector.yolo_facedetection(rgb)
    elif mode == 'dlib-cnn':
        img = fa_detector.dlibcnn_facedetection(rgb)
    end_time = time.time() - st_time

    cv2.imshow('Out Img',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    print("TIME: " + str(end_time))

# if main function
if __name__ == '__main__':

    fin = sys.argv[1]
    test_getface(fin,mode='cv2dnn')











