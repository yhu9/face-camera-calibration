
import os

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import cv2

import util

class DataLoader():
    def __init__(self):
        self.dataroot = '/home/huynshen/data/kitti_data_object_image_2'
        self.dataset = 'training'
        self.datadir = os.path.join(self.dataroot,self.dataset)
        self.img_filename = 'image_2'
        self.calib_filename = 'calib'
        self.label_filename = 'label_2'

        self.image_paths = self.getImageFileNames()
        self.calib_paths = self.getCalibFileNames()
        self.label_paths = self.getLabelFileNames()

    def getImageFileNames(self):
        data_path = os.path.join(self.datadir,self.img_filename)
        fnames = os.listdir(data_path)
        full_paths = [os.path.join(data_path,f) for f in fnames]
        full_paths.sort()
        return full_paths

    def getCalibFileNames(self):
        data_path = os.path.join(self.datadir,self.calib_filename)
        fnames = os.listdir(data_path)
        full_paths = [os.path.join(data_path,f) for f in fnames]
        full_paths.sort()
        return full_paths

    def getLabelFileNames(self):
        data_path = os.path.join(self.datadir,self.label_filename)
        fnames = os.listdir(data_path)
        full_paths = [os.path.join(data_path,f) for f in fnames]
        full_paths.sort()
        return full_paths

    # return image as numpy array in rgb [0,255]
    def readImageFile(self,f):
        img = imread(f)
        return img

    # return 3x4 calibration matrix as numpy array
    def readCalibFile(self,f):
        with open(f,'r') as fin:
            lines = fin.readlines()
            l = lines[0]
            tokens = l.split(' ')
            K = np.array(tokens[1:]).astype(np.float32)
        return K.reshape((3,4))

    # return dictionary with
    # 2d bounding box information
    # 3d bounding box information
    def readLabelFile(self,f):
        data = []
        with open(f,'r') as fin:
            lines = fin.readlines()
            for l in lines:
                instance = {}
                tokens = l.split(' ')
                instance['label'] = tokens[0]
                if instance['label'] == 'DontCare': break
                instance['truncation'] = float(tokens[1])
                instance['occlusion'] = float(tokens[2])
                instance['alpha'] = float(tokens[3])
                instance['bbox2d'] = (float(tokens[4]),
                        float(tokens[5]),
                        float(tokens[6]),
                        float(tokens[7]))   # left top right bottom
                instance['bbox3d'] = (float(tokens[8]), # box width
                        float(tokens[9]), # box height
                        float(tokens[10]), #box length
                        float(tokens[11]), # location x
                        float(tokens[12]), # location y
                        float(tokens[13]), # location z
                        float(tokens[14])) # yaw angle
                data.append(instance)
        return data

    # read n samples and construct corresponding 3d corner pts and 2d pt pairs
    def readN(self,n):

        wpts = np.empty((n,8,3))
        uvpts = np.empty((n,8,2))
        i = 0
        for f1,f2,f3 in zip(self.image_paths,self.label_paths,self.calib_paths):
            data = self.readLabelFile(f2)
            for instance in data:
                if instance['label'] == '': continue
                #img = self.readImageFile(f1)
                calib = self.readCalibFile(f3)
                print(calib)

                l,t,r,b = data[0]['bbox2d']
                h,w,l,x,y,z,ry = data[0]['bbox3d']

                p3d = util.get3DbboxPts(l,w,h,x,y,z,ry)
                p2d = util.projectToImage(p3d,calib)

                wpts[i] = p3d.T
                uvpts[i] = p2d.T[:,:2]
            i+= 1
            if i == n: break

        return uvpts, wpts

# Test out the dataloader
if __name__ == '__main__':

    loader = DataLoader()

    for f1,f2,f3 in zip(loader.image_paths,loader.label_paths,loader.calib_paths):
        img = loader.readImageFile(f1)
        data = loader.readLabelFile(f2)
        calib = loader.readCalibFile(f3)

        l,t,r,b = data[0]['bbox2d']
        h,w,l,x,y,z,ry = data[0]['bbox3d']

        p3d = util.get3DbboxPts(l,w,h,x,y,z,ry)
        p2d = util.projectToImage(p3d,calib)

        img = util.drawConvexHull(p2d.T[:,:2],img)

        plt.imshow(img)
        plt.show()

        quit()
