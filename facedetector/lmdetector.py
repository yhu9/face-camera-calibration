
import os

import face_alignment
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.io as scipy

from detect import FADetector

#########################################
class LMDetector():

    def __init__(self):

        # adrian bulat's landmark detection uses FAN hourglass method. and trains 3D detector on AFLW 3D data
        # it comes with its own face detector, and it may need some fiddeling to remove
        self.fa_net2D = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False)
        self.fa_net3D = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,flip_input=False)
        self.pred2d = None
        self.pred3d = None

    # assume only one face in the image
    def detect2D(self,input):
        preds = self.fa_net2D.get_landmarks(input)
        self.pred2d = preds[-1]
        return preds

    # assume only one face in the image
    def detect3D(self,input):
        preds = self.fa_net3D.get_landmarks(input)
        curr = 0
        for idx,lmset in enumerate(preds):
            height = abs(min(lmset[:,0]) - max(lmset[:,0]))
            width = abs(min(lmset[:,1]) - max(lmset[:,1]))
            cond = curr < height*width
            if cond:
                maxidx = idx
                curr = height*width

        self.pred3d = preds[0]
        return self.pred3d

    def show3d_lm(self,canvas,color=[255,0,0]):
        if self.pred3d is None: return None

        h,w,d = canvas.shape
        minu,maxu = np.amin(self.pred3d[:,0]),np.amax(self.pred3d[:,0])
        minv,maxv = np.amin(self.pred3d[:,1]),np.amax(self.pred3d[:,1])
        img = np.zeros((max(int(maxv),h),max(int(maxu),w),d))
        img[:h,:w] = canvas

        for pt in self.pred3d:
            cv2.circle(img,(pt[0],pt[1]),2,color,1)
        cv2.imshow('f',cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        return img.astype(np.uint8)

    def show2d_lm(self,canvas,color=[255,255,255]):
        if self.pred2d is None: return None

        h,w,d = canvas.shape
        minu,maxu = np.amin(self.pred2d[:,0]),np.amax(self.pred2d[:,0])
        minv,maxv = np.amin(self.pred2d[:,1]),np.amax(self.pred2d[:,1])
        img = np.zeros((max(maxv,h),max(maxu,w),d))
        img[:h,:w] = canvas

        for pt in self.pred3d:
            cv2.circle(img,(pt[0],pt[1]),6,color,-1)
        cv2.imshow('f',cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        return img

# process a single file
def processfile(img,fa,mode='3d'):

    if mode == '3d':
        lm_3d = fa.detect3D(img)
        img = fa.show3d_lm(img)
        return img, lm_3d
    elif mode == '2d':
        lm_2d = fa.detect2D(img)
        fa.show2d_lm(img.copy())
        return img, lm_2d

# process an entire directory
def processpath(filepath,outfile,fa, save=True):
    face_detector = FADetector()

    if os.path.isdir(filepath):
        filenames = [os.path.join(filepath,f) for f in os.listdir(filepath)]
    else:
        print('error with processpath function')
        return

    imgs = []
    landmarks3d = []
    for f in filenames:
        print(f"processing: {f}")
        img = io.imread(f)
        bbox = face_detector.cv2dnn_facedetection(img)
        if bbox is None: continue
        else:
            canvas = img.copy()
            pad = 40
            canvas[:int(bbox['top'])-pad] *= 0
            canvas[int(bbox['bottom'])+pad:] *= 0
            canvas[:,:int(bbox['left'])-pad] *= 0
            canvas[:,int(bbox['right'])+pad:] *= 0

        canvas,lm3d = processfile(canvas,fa)
        img[canvas != 0] = canvas[canvas != 0]

        if save:
            plt.imsave(os.path.join(outfile,os.path.basename(f)),img)

        imgs.append(f)
        landmarks3d.append(lm3d)

    # get one 2d-3d landmark pair for one face in each image
    data = {}
    lm3d = np.stack(landmarks3d,axis=-1)
    imgs = np.stack(imgs)
    data['lm3d'] = lm3d
    data['imgs'] = imgs
    scipy.savemat(os.path.join(outfile,'lm3d.mat'),data)

    return np.stack(landmarks3d,axis=-1), np.stack(imgs)

#######################################
# unit test
if __name__ == '__main__':
    import option
    args = option.args

    # create our landmark detector
    fa = LMDetector()

    # process path to image or directory of images
    if os.path.isdir(args.path):
        outfile = os.path.join(args.outpath,os.path.basename(args.path))

        if not os.path.exists(args.outpath): os.mkdir(args.outpath)
        if not os.path.exists(outfile): os.mkdir(outfile)
        lm3d,imgs = processpath(args.path,outfile,fa)

    # process a single file for visualization
    elif os.path.isfile(args.path):
        img = io.imread(args.path)
        face_detector = FADetector()
        bbox = face_detector.cv2dnn_facedetection(img)
        if bbox is None: print('error! no face detected')
        else:
            canvas = img.copy()
            pad = 40
            canvas[:int(bbox['top'])-pad] *= 0
            canvas[int(bbox['bottom'])+pad:] *= 0
            canvas[:,:int(bbox['left'])-pad] *= 0
            canvas[:,int(bbox['right'])+pad:] *= 0

        canvas,lm3d = processfile(canvas,fa)
        print(lm3d.shape)
        quit()
        img[canvas != 0] = canvas[canvas != 0]

        data = {}
        data['sample_lm3d'] = lm3d
        data['img'] = img
        scipy.savemat('output/sample.mat',data)

        cv2.waitKey(1)




