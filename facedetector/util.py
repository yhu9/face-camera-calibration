
import math
import os
import sys
import xml.etree.ElementTree as ET
import struct
import time
import random

import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
import pptk

def decompose_projection_matrix(P):
        p1 = P[:, 0]
        p2 = P[:, 1]
        p3 = P[:, 2]
        p4 = P[:, 3]
        M = np.zeros((3, 3))
        M[:, 0] = p1
        M[:, 1] = p2
        M[:, 2] = p3
        m3 = np.transpose(M[2, :])
        tmp = np.zeros((3, 3))
        tmp[:, 0] = p2
        tmp[:, 1] = p3
        tmp[:, 2] = p4
        X = np.linalg.det(tmp)
        tmp = np.zeros((3, 3))
        tmp[:, 0] = p1
        tmp[:, 1] = p3
        tmp[:, 2] = p4
        Y = -np.linalg.det(tmp)
        tmp = np.zeros((3, 3))
        tmp[:, 0] = p1
        tmp[:, 1] = p2
        tmp[:, 2] = p4
        Z = np.linalg.det(tmp)
        tmp = np.zeros((3, 3))
        tmp[:, 0] = p1
        tmp[:, 1] = p2
        tmp[:, 2] = p3
        T = -np.linalg.det(tmp)
        Pc = np.zeros((4, 1))
        Pc[0] = X
        Pc[1] = Y
        Pc[2] = Z
        Pc[3] = T
        Pc = Pc/Pc[3]
        Pc = Pc[0:3]
        pp = np.matmul(M, m3)
        pp = pp/pp[2]
        pp = pp[0:2]
        pv = np.linalg.det(M)*m3
        pv = pv/np.linalg.norm(pv)
        (K, Rc_w) = RQ3(M)
        #if(np.dot(np.cross(Rc_w[:, 0], Rc_w[:, 1]), Rc_w[:, 2]) < 0):
        #       print("Note that rotation matrix is left handed")
        return K, Rc_w, Pc, pp, pv

# Convert A to RQ decomposition using QR decomposition
# https://math.stackexchange.com/questions/1640695/rq-decomposition/1640762
def QRtoRQ(A):
        np.linalg.qr(A)

# RQ decomposition of A
def RQ3(A):
        dims = np.shape(A)
        if(dims[0] != 3 or dims[1] != 3):
                print("A must be 3x3")
        eps = 10**-10
        A[2, 2] = A[2, 2]+eps
        c = -A[2, 2]/math.sqrt(A[2, 2]**2+A[2, 1]**2)
        s = A[2, 1]/math.sqrt(A[2, 2]**2+A[2, 1]**2)
        Qx = np.zeros((3, 3))
        Qx[0, 0] = 1
        Qx[1, 1] = c
        Qx[1, 2] = -s
        Qx[2, 1] = s
        Qx[2, 2] = c
        R = np.matmul(A, Qx)
        R[2, 2] = R[2, 2]+eps
        c = R[2, 2]/math.sqrt(R[2, 2]**2+R[2, 0]**2)
        s = R[2, 0]/math.sqrt(R[2, 2]**2+R[2, 0]**2)
        Qy = np.zeros((3, 3))
        Qy[0, 0] = c
        Qy[0, 2] = s
        Qy[1, 1] = 1
        Qy[2, 0] = -s
        Qy[2, 2] = c
        R = np.matmul(R, Qy)
        R[1, 1] = R[1, 1]+eps
        c = -R[1, 1]/math.sqrt(R[1, 1]**2+R[1, 0]**2)
        s = R[1, 0]/math.sqrt(R[1, 1]**2+R[1, 0]**2)
        Qz = np.zeros((3, 3))
        Qz[0, 0] = c
        Qz[0, 1] = -s
        Qz[1, 0] = s
        Qz[1, 1] = c
        Qz[2, 2] = 1
        R = np.matmul(R, Qz)
        Q = np.matmul(np.matmul(np.transpose(Qz), np.transpose(Qy)), np.transpose(Qx))
        for n in range(3):
            if(R[n, n] < 0):
                R[:, n] = -R[:, n]
                Q[n, :] = -Q[n, :]

        return R,Q

def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
        assert(isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
                x = math.atan2(R[2,1] , R[2,2])
                y = math.atan2(-R[2,0], sy)
                z = math.atan2(R[1,0], R[0,0])
        else :
                x = math.atan2(-R[1,2], R[1,1])
                y = math.atan2(-R[2,0], sy)
                z = 0
        return np.array([x, y, z])


def getRyMatrix(ry):
    R = np.array([[np.cos(ry),0,np.sin(ry)],
        [0,1,0],
        [-np.sin(ry),0,np.cos(ry)]])
    return R

def get3DbboxPts(l,w,h,x,y,z,ry):
    x_corners = np.array([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2])
    y_corners = np.array([0,0,0,0,-h,-h,-h,-h])
    z_corners = np.array([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2])

    R = getRyMatrix(ry)
    pts = np.stack((x_corners,y_corners,z_corners),axis=0)
    rot_pts = R.dot(pts)
    rot_pts = rot_pts + np.expand_dims(np.array([x,y,z]),axis=1)

    return rot_pts

def projectToImage(p3d,K):
    homogeneous_pts = np.concatenate((p3d,np.ones((1,p3d.shape[1]))),axis=0)
    p2d = K.dot(homogeneous_pts)
    p2d = p2d / np.expand_dims(p2d[-1],axis=0)
    return p2d

def drawConvexHull(p2d,img):
    ctr = p2d.reshape((-1,1,2)).astype(np.int32)
    hull = cv2.convexHull(ctr,True)
    cv2.drawContours(img,[hull],0,[0,1,0],2)
    return img

def drawpts(img, pts, color=[255,255,255]):
    for p in pts:
        cv2.circle(img,(int(p[0]),int(p[1])),1,color,-1)

    return img

# convert points to a homogeneous coordinate system
def convertHomogeneous(pts):
    return

# read biwi kinect head pose data
def read_pose(file_in):

    pose = []
    center = []
    with open(file_in,'r') as fin:

        lines = fin.readlines()
        pose.append([float(val) for val in lines[0].split()])
        pose.append([float(val) for val in lines[1].split()])
        pose.append([float(val) for val in lines[2].split()])
        center = [float(val) for val in lines[4].split()]

    pose = np.array(pose)

    return np.array(pose), np.array(center)

def readBIWICalibration(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
        K = [lines[0].split(), lines[1].split(),lines[2].split()]
        K = np.array(K,dtype=np.float32)

    return K

def createHeatMap(matrix, xlabels='', ylabels='', title="heat map", xtitle="x axis", ytitle="y axis", save=False,fileout='out.png'):

    N,M = matrix.shape
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect='auto')

    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))

    if xlabels != '':
        ax.set_xticklabels(xlabels)
    if ylabels != '':
        ax.set_yticklabels(ylabels)


    plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    for i in range(N):
        for j in range(M):
            text = ax.text(j,i,np.around(matrix[i,j],decimals=2), ha="center", va="center", color="w",fontsize=10)

    #plt.tight_layout()
    ax.set_title(title)
    fig.tight_layout()
    if not save:
        plt.show()
    else:
        fig.savefig(fileout)

    plt.close(fig)

# get euler angles in pitch: [-90,90]
# get euler angles in yaw: [-90,90]
# get euler angles in roll: [-180,180]
# positive yaw will have face look left
# positive pitch will have face look up
# positive roll will have face roll counter clockwise
def getEulerAngles(R):

    # https://www.gregslabaugh.net/publications/euler.pdf
    # two possible solutions for ry (yaw). Numpy restricts arcsin to [-90,90] degrees
    # so we only take solutions within this range for ry. however for completeness we include the
    # other solutions as comments
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html
    if not ((R[2,0] >= 0.999 and R[2,0] <= 1.0001) or (R[2,0] <= -0.999 and R[2,0] >= -1.0001)):
        ry = -np.arcsin(R[2,0])
        #ry2 = np.pi - ry1
        rx = np.arctan2(R[2,1]/np.cos(ry), R[2,2]/np.cos(ry))
        #rx2 = np.arctan2(R[2,1]/np.cos(ry2), R[2,2]/np.cos(ry2))
        rz = np.arctan2(R[1,0]/np.cos(ry), R[0,0]/np.cos(ry))
        #rz2 = np.arctan2(R[1,0]/np.cos(ry2), R[0,0]/np.cos(ry2))
    else:
        rx = 0
        if (R[2,0] <= -0.999 and R[2,0] >= -1.0001):
            ry = np.pi/2
            rz = rx + np.arctan2(R[0,1],R[0,2])
        else:
            ry = -np.pi/2
            rz = -rx + np.arctan2(-R[0,1],-R[0,2])
    return rx,ry,rz

# convert radians to degrees
def convertRadiansToDegrees(rad):
    return (rad * 180) / np.pi

# get ground truth 2d landmark locations for this subject
def readXML(xmlfile):
    tree = ET.parse(xmlfile)
    images = tree.findall('images/image')
    rootdir = '/home/huynshen/data/kinect_head_pose_db/biwi_max30'
    subject = f"{int(os.path.basename(xmlfile)[:-4]):02d}"
    subject_dir = os.path.join(rootdir,subject)
    M = len(images); N = 21
    lm2d = np.ones((M,N,3)) * -1
    files = []

    for i,img in enumerate(images):
        full_path = os.path.join(subject_dir,img.attrib['file'])
        if os.path.exists(full_path):
            box = img[0]
            files.append(full_path)
            pts = box.findall('part')
            for p in pts:
                idx = int(p.attrib['name'])
                xval = int(p.attrib['x'])
                yval = int(p.attrib['y'])
                lm2d[i,idx,0] = xval
                lm2d[i,idx,1] = yval
                lm2d[i,idx,2] = 1
        else:
            print(f"error reading {full_path}")

    return lm2d, np.array(files)

def readBIWICalib(calib_file):
    with open(calib_file,'r') as fin:
        lines = fin.readlines()
        K = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
        R = np.array([[float(x) for x in lines[i].split()] for i in range(6,9)])
        T = np.array([[float(x) for x in lines[10].split()]])

    return K, R, T

#PYTHON VERSION FOR LOADING .BIN FILE OF THE COMPRESSED KINECT DATASET FROM (NOTE VISUALIZATION IS PROVIDED IN THE UNIT TEST BELOW)
#THE BIWI KINECT DATASET
#WE USE THE BIWI KINECT DATASET AS OUR LOW RESOLUTION DATA
#in  <- file name
#out:   m * n depth map
#       pts3d (K,3) pts in camera coordinates (mm)
#       pts2d (K,2) pts in image coordinates (pixels)
def readDepthData(fname,rgb_calib,d_calib):

    #READ DEPTH BINARY FILE
    with open(fname,'rb') as depth_file:

        im_width = struct.unpack('i',depth_file.read(4))[0]         #integers are 32 bits or 4 bytes each. use struct unpack to determine the integer value
        im_height = struct.unpack('i',depth_file.read(4))[0]

        depth_img = [None] * (im_width * im_height)
        p = 0
        while(p < im_width * im_height):
            numempty = struct.unpack('i',depth_file.read(4))[0]
            for i in range(numempty):
                depth_img[p + i] = 0;
            numfull = struct.unpack('i',depth_file.read(4))[0]

            #depth information is stored as 16bit integers or int16_t. There is a possibility
            #I may have implemented this incorrectly
            for i in range(numfull):
                depth_img[p+numempty+i] = struct.unpack('<h',depth_file.read(2))[0]         #define little endian file data organization

            p += numempty+numfull

    #READ THE RGB CALIBRATION FILE
    rgb_I, rgb_R, rgb_T2 = readBIWICalib(rgb_calib)

    #DEPTH MAP CALIBRATION FILE
    d_I, d_R, d_T2 = readBIWICalib(d_calib)

    #APPLY CALIBRATION
    k_inv = np.linalg.inv(d_I)
    dmap = np.array(depth_img).reshape((im_height,im_width))
    dimg = np.dstack(np.meshgrid(np.arange(im_width),np.arange(im_height)) + [np.ones(dmap.shape)])
    pts = dimg[dmap != 0]
    proj = k_inv.dot(pts.T)
    xyz = np.expand_dims(dmap[dmap!=0],axis=0) * proj
    proj_ = rgb_R.dot(xyz) + rgb_T2.T
    rgb_proj = rgb_I.dot(proj_)
    rgb_proj = rgb_proj/np.expand_dims(rgb_proj[-1,:],axis=0)

    canvas = np.ones((im_height,im_width,2)) * -1
    for i, p in enumerate(rgb_proj.T):
        p = p.astype(np.uint16)
        if p[1] < im_height and p[0] < im_width:
            if canvas[p[1],p[0],0] == -1 or (canvas[p[1],p[0],0] != 0 and canvas[p[1],p[0],0] > proj_.T[i,-1]):
                canvas[p[1],p[0],0] = proj_.T[i,-1]
                canvas[p[1],p[0],1] = i

    pts3d = proj_
    pts2d = rgb_proj

    #canvas = canvas / np.amax(canvas)
    #plt.imshow(canvas)
    #plt.show()

    # reluctant to get rid of this since it is faster, but doesn't consider overlapping depth
    # along edges. Maybe its not a problem?
    '''
    dimg = np.array(depth_img).reshape((im_height,im_width))
    dimg = np.dstack(np.meshgrid(np.arange(im_width),np.arange(im_height)) + [dimg])
    points = dimg.reshape((im_height * im_width),3).astype(np.float32)

    #FIND 3D COORD OF DEPTH MAP USING INTRINSIC CAMERA VALUES OF DEPTH CALIBRATION FILE
    cx_d = d_I[0][2]       #INTRINSIC CAMERA VALUES
    cy_d = d_I[1][2]
    fx_d = d_I[0][0]
    fy_d = d_I[1][1]
    #https://stackoverflow.com/questions/22587178/how-to-convert-kinect-raw-depth-info-to-meters-in-matlab
    points[:,0] = (points[:,0] - cx_d) * points[:,2] / fx_d     #world 3d x point
    points[:,1] = (points[:,1] - cy_d) * points[:,2] / fy_d     #world 3d y point
    points[:,2] = points[:,2]

    #FIND 2D COORD OF DEPTH MAP USING INTRINSIC CAMERA VALUES OF RGB CALIBRATION FILE
    cx_r = rgb_I[0][2]
    cy_r = rgb_I[1][2]
    fx_r = rgb_I[0][0]
    fy_r = rgb_I[1][1]
    points = np.matmul(rgb_R,np.transpose(points))
    points = np.transpose(points)
    points = points + rgb_T2
    points[:,0] = (points[:,0] * fx_r / points[:,2]) + cx_r
    points[:,1] = (points[:,1] * fy_r / points[:,2]) + cy_r
    points = points[points[:,0] >= 0]
    points = points[points[:,1] >= 0]
    points = points[points[:,0] < im_width]
    points = points[points[:,1] < im_height]
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    img = np.zeros((im_height,im_width))
    img[(Y.astype(np.uint32)),(X.astype(np.uint32))] = Z

    pptk.viewer(points)
    plt.imshow(img)
    plt.show()
    '''

    return canvas, pts3d, pts2d

# random walk on mask until truth value is found. May never terminate...
# expects a 2D truth mask and a (x,y) location as a tuple
# perhaps replace with true randomness
def randomWalk(mask, loc=(0,0), maxiter=100):
    h,w = mask.shape
    if mask[loc[1],loc[0]]:
        return loc
    else:
        x,y = loc
        x = int(x); y = int(y)
        while not mask[y,x]:
            bit1 = random.randint(0,2)
            bit2 = random.randint(0,2)
            bit3 = random.randint(0,2)
            if bit1 == 0 and bit2 == 0 and bit3 == 0:
                y += 1
            elif bit1 == 0 and bit2 == 0 and bit3 == 1:
                y -= 1
            elif bit1 == 0 and bit2 == 1 and bit3 == 0:
                x += 1
            elif bit1 == 0 and bit2 == 1 and bit3 == 1:
                x -= 1
            elif bit1 == 1 and bit2 == 0 and bit3 == 0:
                x += 1; y += 1
            elif bit1 == 1 and bit2 == 0 and bit3 == 1:
                x += 1; y -= 1
            elif bit1 == 1 and bit2 == 1 and bit3 == 0:
                x -= 1; y += 1
            elif bit1 == 1 and bit2 == 1 and bit3 == 1:
                x -= 1; y -= 1

            if y == h: y -= 1
            if y == -h: y += 1
            if x == w: x -= 1
            if x == -w: x += 1

    return (x,y)

# find the nearest point to a location that is valid in the mask
def findValidPoint(mask, loc=(0,0),maxiter=100):
    h,w = mask.shape
    if mask[loc[1],loc[0]]:
        return loc
    else:
        s = 0
        found = False
        while not found:
            s += 1
            top = max(loc[1]-s,0)
            bot = min(loc[1]+s+1,h)
            left = max(loc[0]-s,0)
            right = min(loc[0]+s+1,w)
            found = np.any(mask[top:bot,left:right])
        submask = mask[top:bot,left:right]
        row, col = np.where(submask)

        y = max(0,loc[1] - s) + row[0]
        x = max(0,loc[0] - s) + col[0]

        return (x,y)

# get accurate eye to eye distance data with biwi dataset pred 2d landmarks
def getFaceDataBIWI_gtsize(subject, maxangle=20, lmrange=(0,68)):
    if lmrange == (0,68):
        leye, reye = 36, 46
    elif lmrange == (27,68) or lmrange == (27,48):
        leye, reye = 9, 19

    # get pred 2d landmark locations
    uvcoord, worldpts, valid_paths = getFaceDataBIWI(subject,maxangle=maxangle,lmrange=lmrange)

    # get ground truth landmark locations of uvcoord in camera coordinate system
    # by looking at the depth
    root_dir = f'/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}'
    M = len(valid_paths)
    N = lmrange[1] - lmrange[0]
    worldcoord = np.ones((M,N,4))
    for i,f in enumerate(valid_paths):
        xyzfile = f.replace('rgb','xyz').replace('png','mat').replace('hpdb_copy','hpdb')

        # load the data kinect data with camera info
        data = scipy.io.loadmat(xyzfile)
        p3d = data['pts3d']
        p2d = data['pts2d']
        dmap = data['dmap']

        mask = dmap[:,:,-1] != -1
        lm2d = uvcoord[:,:,i]
        pt1 = lm2d[leye]
        pt2 = lm2d[reye]

        # sanity check
        #img = plt.imread(rgbfile)
        #img = drawpts(img, lm2d)
        #plt.imshow(img)
        #plt.show()

        # random walk for sparse dmap to find a point close to or on desired location
        x1,y1 = findValidPoint(mask, (int(pt1[0]),int(pt1[1])))
        x2,y2 = findValidPoint(mask, (int(pt2[0]),int(pt2[1])))
        idx1 = dmap[y1,x1][-1]
        idx2 = dmap[y2,x2][-1]

        left_eye = p3d[:,int(idx1)]
        right_eye = p3d[:,int(idx2)]
        distance = np.linalg.norm(left_eye - right_eye)

        # adjust 3d shape to fit to eye to eye distance in camera coordinate space
        ratio = distance / np.linalg.norm(worldpts[leye,:] - worldpts[reye,:])
        worldcoord[i,:,:3] = worldpts[:,:3] * ratio

    return uvcoord, worldcoord, valid_paths

# get weak perspective landmarks from biwi along with 3d landmarks
def getFaceDataBIWI_weak(subject, maxangle=20, lmrange=(0,68)):
    # load 3d pt data
    lm3d_path = f"../matlab/sub{subject}_meu3d.mat"
    data3d = scipy.io.loadmat(lm3d_path)
    worldpts = data3d['meu3d']

    # load 2d pt via weak projection alignment of 3d data
    #lm2d_path = f"/home/huynshen/projects/face-camera-calibration/facedetector/result/biwi_kinect_3dlm_prediction/{subject}/lm3d.mat"
    lm2d_path = f"/home/huynshen/projects/face-camera-calibration/facedetector/tmp/{subject}/lm3d.mat"
    data2d = scipy.io.loadmat(lm2d_path)
    lm2d = data2d['lm3d']
    valid_paths = data2d['imgs']

    # get head pose information per file with centers
    thetax = []
    thetay = []
    thetaz = []
    centers = []
    for p in valid_paths:
        p = p.replace('hpdb_copy', 'hpdb')
        p = p.replace('rgb', 'pose')
        p = p.replace('png', 'txt')
        R, center = read_pose(p)
        rx,ry,rz = [convertRadiansToDegrees(theta) for theta in getEulerAngles(R)]
        thetax.append(rx)
        thetay.append(ry)
        thetaz.append(rz)
        centers.append(center)
    thetax = np.stack(thetax,axis=0)
    thetay = np.stack(thetay,axis=0)
    thetaz = np.stack(thetaz,axis=0)
    centers = np.stack(centers,axis=0)

    # restrict large poses greater than max degrees (default 20)
    valid_pose = ((np.absolute(thetax) <= maxangle) &
        (np.absolute(thetay) <= maxangle) &
        (np.absolute(thetaz) <= maxangle))
    count = np.sum(valid_pose.astype(np.float32))
    print("number of valid poses: ", count)

    valid_paths = valid_paths[valid_pose]
    lm2d = lm2d[:,:,valid_pose]

    # set output containers in homegeneous coordinates with wpts scaled and assigned landmarks
    N = lmrange[1] - lmrange[0]
    uvcoord = lm2d[lmrange[0]:lmrange[1]]
    uvcoord[:,-1] = 1
    worldcoord = np.ones((N,4))
    ratio = 122 / np.linalg.norm(worldpts[36,:] - worldpts[46,:])
    worldpts = worldpts * ratio
    worldcoord[:,:3] = worldpts[lmrange[0]:lmrange[1]]

    return uvcoord, worldcoord

# get ground truth 2d to 3d landmarks by projecting 2d landmarks to 3d using known intrinsics and depth
def getFaceDataBIWI_gt(subject, valid_paths,maxangle=20, lmrange=(0,68)):

    # if I wanted to load every single predicted 3d object
    # lm3d_path = f"/home/huynshen/projects/face-camera-calibration/facedetector/result/biwi_kinect_3dlm_prediction/{subject}/lm3d.mat"
    # lm3d_data = scipy.io.loadmat(lm3d_path)['lm3d']

    # get 3d world points
    lm3d_path = f"../matlab/sub{subject}_meu3d.mat"
    data3d = scipy.io.loadmat(lm3d_path)
    worldpts = data3d['meu3d']

    # get processed gt 2d points of landmarks
    lm2d_dir = f"/home/huynshen/projects/face-camera-calibration/facedetector/result/biwi_kinect_gtlm2d/{subject}/"
    M = len(valid_paths)
    N = lmrange[1] - lmrange[0]
    uvcoord = np.ones((N,3,M))
    for i, f in enumerate(valid_paths):
        f = os.path.basename(f.replace('rgb','gtlm2d').replace('png','mat'))
        full_path = os.path.join(lm2d_dir,f)
        data2d = scipy.io.loadmat(full_path)
        uvcoord[:,:,i] = data2d['lm2d'][lmrange[0]:lmrange[1]]

    # convert world coord to homogeneous coordinate system
    # and resize by single scale according to average intraocular distance
    ratio = 122 / np.linalg.norm(worldpts[36,:] - worldpts[46,:])
    worldpts = worldpts * ratio
    worldcoord = np.ones((N,4))
    worldcoord[:,:3] = worldpts[lmrange[0]:lmrange[1]] * ratio

    return uvcoord, worldcoord

# get synthetic data under weak perspective projection on subject
def getFaceDataBIWI_synth(subject, maxangle=20, lmrange=(0,68)):

    # load matlab files containt world and
    lm3d_path = f"../matlab/sub{subject}_meu3d.mat"
    data3d = scipy.io.loadmat(lm3d_path)
    worldpts = data3d['meu3d']

    root_dir = f'/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}'
    files = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f[-4:] == '.mat']
    files.sort()
    for f in files:
        data = scipy.io.loadmat(f)

    img = plt.imread(rgb_file)
    dmap = dmap / np.amax(dmap)
    img[:,:,-1] = dmap
    img[dmap == 0] = 0
    plt.imshow(img)
    plt.show()

    K_gt = readBIWICalibration(calib_file)

# more realistic dataset using 3d landmark locations with corresponding 2d locations
# found via face-alignment method
def getFaceDataBIWI(subject,maxangle=20, lmrange=(0,68)):
    lm2d_path = f"/home/huynshen/projects/git/landmark-detection/SAN/tmp/biwi_kinect_2dlm/{subject}/lm2d.mat"
    lm3d_path = f"../matlab/sub{subject}_meu3d.mat"

    # load the data
    # data[lm2d] provides confidence values along with each landmark point
    data2d = scipy.io.loadmat(lm2d_path)
    lm2d = data2d['lm2d']
    data3d = scipy.io.loadmat(lm3d_path)
    worldpts = data3d['meu3d']

    # image file paths
    imgpaths = data2d['files']

    # get head pose information per file with centers
    thetax = []
    thetay = []
    thetaz = []
    centers = []
    for p in imgpaths:
        p = p.replace('hpdb_copy', 'hpdb')
        p = p.replace('rgb', 'pose')
        p = p.replace('png', 'txt')
        R, center = read_pose(p)
        rx,ry,rz = [convertRadiansToDegrees(theta) for theta in getEulerAngles(R)]
        thetax.append(rx)
        thetay.append(ry)
        thetaz.append(rz)
        centers.append(center)
    thetax = np.stack(thetax,axis=0)
    thetay = np.stack(thetay,axis=0)
    thetaz = np.stack(thetaz,axis=0)
    centers = np.stack(centers,axis=0)

    # view histogram of angle distributions in pitch yaw and roll
    # plotHistogram(thetax)
    # plotHistogram(thetay)
    # plotHistogram(thetaz)

    # restrict large poses greater than max degrees (default 20)
    valid_pose = ((np.absolute(thetax) <= maxangle) &
        (np.absolute(thetay) <= maxangle) &
        (np.absolute(thetaz) <= maxangle))
    count = np.sum(valid_pose.astype(np.float32))
    print("number of valid poses: ", count)

    # set 2d landmark data restricted to valid views according to pose
    valid_paths = imgpaths[valid_pose]
    lm2d = lm2d[valid_pose,lmrange[0]:lmrange[1],:]
    M, N, _ = lm2d.shape

    # 122 is the average intraocular distance for a male individual
    # scale does not affect the intrinsic estimation, but does affect extrinsic estimation
    ratio = 122 / np.linalg.norm(worldpts[36, :] - worldpts[46, :])
    worldpts = worldpts * ratio

    uvcoord = np.ones((M, N, 3))
    worldcoord = np.ones((N, 4))

    uvcoord[:, :, :2] = lm2d[:,:,:2]
    worldcoord[:, :3] = worldpts[lmrange[0]:lmrange[1]]
    uvcoord = np.swapaxes(uvcoord,0,2)
    uvcoord = np.swapaxes(uvcoord,0,1)

    return uvcoord, worldcoord, valid_paths

# dataset using 3d, 2d landmark locations found from single network using face alignment
# https://github.com/1adrianb/face-alignment
def getFaceData():
    data = sio.loadmat(args.matfile)
    lm3d = data['lm3d']
    imgs = data['imgs']

    worldpts = sio.loadmat('../matlab/facelm3d.mat')['meu3d']
    uvpts = lm3d[:,:2]
    N, _, M = uvpts.shape

    # 122 is the average intraocular distance for a male individual
    ratio = 122 / np.linalg.norm(worldpts[36, :] - worldpts[46, :])
    worldpts = worldpts * ratio

    uvcoord = np.ones((N, 3, M))
    worldcoord = np.ones((N, 4))

    uvcoord[:, :2, :] = uvpts
    worldcoord[:, :3] = worldpts

    return uvcoord, worldcoord,imgs

# more realistic dataset using 3d landmark locations with corresponding 2d locations that were manually
# annotated by Hong
# found via face-alignment method
def getFaceDataBIWI_manual(subject,maxangle=30):
    lm2d_path = f"/home/huynshen/data/kinect_head_pose_db/biwi_max30/{subject}.xml"
    lm3d_path = f"../matlab/sub{subject}_meu3d.mat"

    lm2d, imgpaths = readXML(lm2d_path)

    # load the data
    # data[lm2d] provides confidence values along with each landmark point
    data3d = scipy.io.loadmat(lm3d_path)
    worldpts = data3d['meu3d']

    # get head pose information per file with centers
    # maybe we don't need to bother with this on the ground truth
    thetax = []
    thetay = []
    thetaz = []
    centers = []
    for p in imgpaths:
        p = p.replace('biwi_max30', 'hpdb')
        p = p.replace('rgb', 'pose')
        p = p.replace('png', 'txt')
        R, center = read_pose(p)
        rx,ry,rz = [convertRadiansToDegrees(theta) for theta in getEulerAngles(R)]
        thetax.append(rx)
        thetay.append(ry)
        thetaz.append(rz)
        centers.append(center)
    thetax = np.stack(thetax,axis=0)
    thetay = np.stack(thetay,axis=0)
    thetaz = np.stack(thetaz,axis=0)
    centers = np.stack(centers,axis=0)

    # restrict large poses greater than max degrees (default 20)
    valid_pose = ((np.absolute(thetax) <= maxangle) &
        (np.absolute(thetay) <= maxangle) &
        (np.absolute(thetaz) <= maxangle))
    count = np.sum(valid_pose.astype(np.float32))
    print("number of valid poses: ", count)

    # set 2d landmark data restricted to valid views according to pose
    valid_paths = imgpaths[valid_pose]
    lm2d = lm2d[valid_pose]
    M, N, _ = lm2d.shape

    # 122 is the average intraocular distance for a male individual
    # scale does not affect the intrinsic estimation, but does affect extrinsic estimation
    ratio = 122 / np.linalg.norm(worldpts[36, :] - worldpts[46, :])
    worldpts = worldpts * ratio

    uvcoord = np.ones((M, N, 3))
    worldcoord = np.ones((N, 4))
    uvcoord[:, :, :2] = lm2d[:,:,:2]
    worldcoord[:, :3] = worldpts[27:48]
    uvcoord = np.swapaxes(uvcoord,0,2)
    uvcoord = np.swapaxes(uvcoord,0,1)
    worldcoord = worldcoord - np.min(worldcoord,axis=0)
    worldcoord[:,-1] = 1

    return uvcoord, worldcoord, valid_paths

# get normalization matrix such that uv1 = normalized uv1
def normalize2d(uvcoord):
    N,d,M = uvcoord.shape
    tmp = uvcoord.swapaxes(1,2)
    tmp = tmp.reshape(M*N,d)
    valid_pts = np.all(tmp >= 0,axis=1)
    pts = tmp[valid_pts]
    mean_translation = np.mean(pts,axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts[:,:2],axis=1))

    nmatrix = np.zeros((3,3))
    nmatrix[:,2] = -scale*mean_translation
    nmatrix[0,0] = scale
    nmatrix[1,1] = scale
    nmatrix[2,2] = 1

    npts = np.matmul(np.expand_dims(nmatrix,axis=0),np.swapaxes(uvcoord,0,2))

    return npts, nmatrix

# normalize 3d points across all views
def normalize3d(pts):
    M,N,d = pts.shape
    tmp = pts.reshape(M*N,d)
    mean_translation = np.mean(tmp,axis=0)
    scale = np.sqrt(3) / np.mean(np.linalg.norm(tmp[:,:3],axis=1))

    nmatrix = np.zeros((4,4))
    nmatrix[:,3] = -scale*mean_translation
    nmatrix[0,0] = scale
    nmatrix[1,1] = scale
    nmatrix[2,2] = scale
    nmatrix[3,3] = 1

    pts = np.swapaxes(pts,1,2)
    npts = np.matmul(np.expand_dims(nmatrix,axis=0), pts)

    return npts, nmatrix


# prot of matlab's procrustes analysis function in numpy
# https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def main():
# unit test read pose and get euler angles
    pose, center = read_pose("/home/huynshen/data/kinect_head_pose_db/hpdb/01/frame_00217_pose.txt")
    rx,ry,rz = getEulerAngles(pose)

    rx = convertRadiansToDegrees(rx)
    ry = convertRadiansToDegrees(ry)
    rz = convertRadiansToDegrees(rz)
    print(f"Pitch: {rx}")
    print(f"Yaw: {ry}")
    print(f"Roll: {rz}")
    print("head center: ", center)

# unit test decompose projection matrix
    P = np.zeros((3, 4))
    P[0, 0] = 353.553
    P[0, 1] = 339.645
    P[0, 2] = 277.744
    P[0, 3] = -1449460
    P[1, 0] = -103.528
    P[1, 1] = 23.3212
    P[1, 2] = 459.607
    P[1, 3] = -632525
    P[2, 0] = 0.707107
    P[2, 1] = -0.353553
    P[2, 2] = 0.612372
    P[2, 3] = -918.559
    '''two_row_P = np.load(sys.argv[1])
    print(two_row_P)
    P[0:2, :] = two_row_P
    P[0:3, 2] = np.cross(two_row_P[0:3, 0], two_row_P[0:3, 1])
    P[2, 3] = 1'''
    (K, Rc_w, Pc, pp, pv) = decompose_projection_matrix(P)
    print(K)
    print(Rc_w)
    print(Pc)
    print(pp)
    print(pv)
    euler_angles = rotationMatrixToEulerAngles(Rc_w)
    print(euler_angles)

if __name__ == '__main__':
    #main()

    mask = np.zeros((500,500), dtype=bool)
    mask[275,120] = True
    x,y = findValidPoint(mask,loc=(250,250))

    print(x,y)
