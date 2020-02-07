# native imports
import argparse
import math
import os

# opensource imports
import cv2
import imageio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
#from scipy.optimize import curve_fit
from lmfit import Minimizer, Parameters, report_fit
import seaborn as sns
from shutil import copyfile, copy

import pptk
from mpl_toolkits.mplot3d import Axes3D

# custom imports
import lmdetector
import dataloader
from option import args
import util

#########################################################################
#########################################################################
#########################################################################

# calibrate based on corresponding uv/world coordinates
def calibrate(uvcoord,worldcoord):
    N,d,M = worldcoord.shape
    H = np.empty((3,3,M))       # up to some scale factor which can be solved trivially

    scales = []
    errors = []
    for i in range(M):
        H[:,:,i],_ = getHomography(uvcoord[:,:,i],worldcoord[:,:,i])
        tmp = H[:,:,i].dot(worldcoord[:,:,i].T).T
        reprojection = tmp/np.expand_dims(tmp[:,-1],axis=-1)
        scales.append(tmp[:,-1])
        error = np.mean(np.linalg.norm(reprojection - uvcoord[:,:,i],axis=1))
        errors.append(error)

    plotHistogram(np.array(errors))
    quit()
    #hist = np.array(scales).flatten()
    #plt.hist(hist,bins=100)
    #plt.show()
    #quit()

    tmp = worldcoord[:,:,i].dot(H[:,:,i].T)
    print(tmp / np.expand_dims(tmp[:,-1],axis=-1))
    print(1/tmp[:,-1])

    B = computeB(H)
    K = getIntrinsics(B)
    R,T = getExtrinsics(K,H)
    Rt = np.concatenate((R,np.expand_dims(T,1)),axis=1)

    return H,K,Rt

def calibrate2(uvcoord,worldcoord):
    N,d,M = worldcoord.shape
    H = np.empty((3,4,M))

    for i in range(M):
        H[:,:,i],_ = getDLT(uvcoord[:,:,i],worldcoord[:,:,i])

    #tmp = H[:,:,i].dot(worldcoord[:,:,i].T).T
    #reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
    #error = np.mean(np.linalg.norm(reprojection[1:] - uvcoord[1:,:,i],axis=1))
    B = computeB12(H)
    K = getIntrinsics(B)
    R,T = getExtrinsics(K,H)
    Rt = np.concatenate((R,np.expand_dims(T,1)),axis=1)

    return H,K,Rt

# find weak perspective projection matrix in rotation, translation, scale using least squares
# input:    2D points = Nx2
#           3D points = Nx3
# output:   s = scale, R = rotation matrix, t = translation vector
# http://www.cs.technion.ac.il/FREDDY/papers/95.pdf
# Reconstruct 2D pts by x = s*R*X + t
# where x is 2d and X is 3D
def computeWeakProjection(pts2d,pts3d):
    # center both set of points
    meu1 = np.mean(pts2d,axis=0)
    meu2 = np.mean(pts3d,axis=0)
    pts1 = pts2d - np.expand_dims(meu1,axis=0)
    pts2 = pts3d - np.expand_dims(meu2,axis=0)
    pts2 = np.concatenate((pts2,np.ones((pts2.shape[0],1))),axis=1)

    N = pts2d.shape[0]
    A = np.zeros((N*2,8))
    A[0::2,0:3] = pts2[:,:3]
    A[0::2,3] = 1
    A[1::2,4:7] = pts2[:,:3]
    A[1::2,7] = 1

    # solve system using least square
    m = np.linalg.lstsq(A,pts1.flatten(),rcond=1)[0]
    m = m.reshape((2,4))
    a = m[0,0:3]
    b = m[1,0:3]

    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    adota = a.dot(a)
    bdotb = b.dot(b)
    adotb = a.dot(b)

    d1 = np.sqrt((adota)*(bdotb)-(adotb)*(adotb))
    d2 = (adota)*(bdotb)+norma*normb*d1-(adotb)*(adotb)
    R = np.zeros((3,3))
    R[0,:] = ((norma+normb) / (2*norma) + normb*adotb*adotb / (2*norma*d2)) * a - adotb * b / (2*d1)
    R[1,:] = ((norma+normb) / (2*normb) + norma * adotb*adotb / (2*normb*d2)) * b - adotb * a / (2*d1)
    s = np.linalg.norm(R[0,:])
    R[0,:] = R[0,:] / s
    R[1,:] = R[1,:] / s
    R[2,:] = np.cross(R[0,:],R[1,:])
    t = meu2.copy()[:3]
    t[-1] = 0
    t = t - np.squeeze(s*R.dot(np.expand_dims(meu2[:3],axis=1)))

    return s,R,t

# get intrinsic and extrinsic params based on optimized H
def getParams(H):
    B = computeB(H)
    K = getIntrinsics(B)
    Rt = getExtrinsics(K,H)
    return K, Rt

# lm3d - 68x3xM
# O(68x3xM) = O(M)
def getCoordinates(lm3d):
    N,d,M = lm3d.shape
    world_coord = np.zeros(lm3d.shape)
    proj = np.zeros(lm3d.shape)

    # zero center the 3d prediction
    meu = np.mean(lm3d,axis=0)
    lm3d_centered = lm3d - meu

    # define a plane for each view
    e = lm3d_centered[33,:,:]
    e = e / np.linalg.norm(e,axis=0)

    # project all 3d points onto a plane for each view
    u = np.expand_dims(np.array([0,0,1]),axis=1)
    I = np.eye(3)
    for i in range(M):
        nvec = np.expand_dims(e[:,i],axis=1)
        proj[:,:,i] = lm3d_centered[:,:,i] - proj[:,:,i].dot(nvec) * e[:,i]
        R = 2 * ((u+nvec).dot((u+nvec).T))/((u+nvec).T.dot((u+nvec))) - I
        world_coord[:,:,i] = proj[:,:,i].dot(R.T)

    # redefine cooresponding uv prediction as points after projection onto plane
    # we set depth to 1 without loss of generality
    uvcoord = (proj + meu)
    uvcoord[:,2,:] = 1

    # set the world coordinate to the mean face of all predictions
    mean_lm = np.mean(world_coord,axis=2)
    world_coord[:,:,:] = np.expand_dims(mean_lm,axis=2)
    s = 146
    d = np.linalg.norm(mean_lm[0,:] - mean_lm[16,:])
    world_coord = world_coord * (s/d)

    return uvcoord, world_coord

# solve for homography matrix using SVD and fundamental constraints of the plane
def getHomography(uv,world):
    N,d = world.shape

    # normalize points
    #tmp = np.concatenate((uv,world),axis=0)
    #mean = np.mean(tmp,axis=0)
    #std = np.std(tmp,axis=0)
    #Nx = np.array([[std[0],0,mean[0]],[0,std[1],mean[1]],[0,0,1]])
    #Nx_inv = np.linalg.inv(Nx)
    #uv = uv.dot(Nx)
    #world = world.dot(Nx)

    # setup constraints
    constraint1 = np.stack((-world[:,0],-world[:,1],-np.ones(N),np.zeros(N),np.zeros(N),np.zeros(N),uv[:,0]*world[:,0],uv[:,0]*world[:,1],uv[:,0]),axis=1)
    constraint2 = np.stack((np.zeros(N),np.zeros(N),np.zeros(N),-world[:,0],-world[:,1],-np.ones(N),uv[:,1]*world[:,0],uv[:,1]*world[:,1],uv[:,1]),axis=1)

    V = np.empty((2*N,9),dtype=np.float32)
    V[0::2,:] = constraint1
    V[1::2,:] = constraint2

    U,S,Vh = np.linalg.svd(V,full_matrices=False)
    h = Vh[-1,:]
    h = h.reshape((3,3))

    tmp = h.dot(world.T).T
    reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
    error = np.mean(np.linalg.norm(reprojection - uv,axis=1))

    return h,error

# linear least square solution to the projection matrix problem
# There are some special cases where this will not work
# https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
def getLinearLeastSquare(uv,world):
    V = np.empty()
    return


# solve for direct linear transformation between world coordinates to uv coordinates up to a scale
def getDLT(uv,world):
    N,d = world.shape

    # setup constraints
    #constraint1 = np.stack((world[:,0],world[:,1],world[:,2],np.ones(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),-uv[:,0]*world[:,0],-uv[:,0]*world[:,1],-uv[:,0]*world[:,2],-uv[:,0]),axis=1)
    #constraint2 = np.stack((np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),world[:,0],world[:,1],world[:,2],np.ones(N),-uv[:,1]*world[:,0],-uv[:,1]*world[:,1],-uv[:,1]*world[:,2],-uv[:,1]),axis=1)
    constraint1 = np.stack((world[:,0],world[:,1],world[:,2],np.ones(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),-uv[:,0]*world[:,0],-uv[:,0]*world[:,1],-uv[:,0]*world[:,2],-uv[:,0]),axis=1)
    constraint2 = np.stack((np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),-world[:,0],-world[:,1],-world[:,2],-np.ones(N),uv[:,1]*world[:,0],uv[:,1]*world[:,1],uv[:,1]*world[:,2],uv[:,1]),axis=1)
    constraint3 = np.stack((uv[:,1]*world[:,0],uv[:,1]*world[:,1],uv[:,1]*world[:,2],uv[:,1],-uv[:,0]*world[:,0],-uv[:,0]*world[:,1],-uv[:,0]*world[:,2],-uv[:,0],np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)),axis=1)

    #V = np.empty((2*N,12),dtype=np.float32)
    #V[0::2,:] = constraint1
    #V[1::2,:] = constraint2
    V = np.empty((3*N,12),dtype=np.float32)
    V[0::3,:] = constraint1
    V[1::3,:] = constraint2
    V[2::3,:] = constraint3

    U,S,Vh = np.linalg.svd(V)
    h = Vh[-1,:]
    h = h.reshape((3,4))

    # tmp = h.dot(world.T).T
    # reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
    # error = np.mean(np.linalg.norm(reprojection[:,1:] - uv[:,1:],axis=1))

    return h

# solve for B using H estimate on each views in order to solve for Intrinsics
def computeB(H):
    _,_,M = H.shape
    V = np.empty((2*M,6))
    constraint1 = np.stack((
        H[0,0,:] * H[0,1,:],
        H[0,0,:] * H[1,1,:] + H[1,0,:] * H[0,1,:],
        H[1,0,:] * H[1,1,:],
        H[2,0,:]*H[0,1,:] + H[0,0,:]*H[2,1,:],
        H[2,0,:]*H[1,1,:] + H[1,0,:]*H[2,1,:],
        H[2,0,:]*H[2,1,:]
        ),axis=1)
    tmp1 = np.stack((
        H[0,0,:]*H[0,0,:],
        H[0,0,:]*H[1,0,:] + H[1,0,:]*H[0,0,:],
        H[1,0,:]*H[1,0,:],
        H[2,0,:]*H[0,0,:] + H[0,0,:]*H[2,0,:],
        H[2,0,:]*H[1,0,:] + H[1,0,:]*H[2,0,:],
        H[2,0,:]*H[2,0,:]
        ),axis=1)
    tmp2 = np.stack((
        H[0,1,:]*H[0,1,:],
        H[0,1,:]*H[1,1,:] + H[1,1,:]*H[0,1,:],
        H[1,1,:]*H[1,1,:],
        H[2,1,:]*H[0,1,:] + H[0,1,:]*H[2,1,:],
        H[2,1,:]*H[1,1,:] + H[1,1,:]*H[2,1,:],
        H[2,1,:]*H[2,1,:]
        ),axis=1)
    constraint2 =  tmp1 - tmp2

    V[0::2,:] = constraint1
    V[1::2,:] = constraint2

    U,S,Vh = np.linalg.svd(V,full_matrices=False)
    B = Vh[-1,:]

    return B

# solve for B using H estimate on each views in order to solve for Intrinsics
# when H is a 3x4xM matrix with M views
def computeB12(H):
    _,_,M = H.shape
    V = np.empty((7*M,6))
    constraint1 = np.stack((
        H[0,0,:]*H[0,1,:],
        H[0,0,:]*H[1,1,:] + H[1,0,:]*H[0,1,:],
        H[1,0,:]*H[1,1,:],
        H[2,0,:]*H[0,1,:] + H[0,0,:]*H[2,1,:],
        H[2,0,:]*H[1,1,:] + H[1,0,:]*H[2,1,:],
        H[2,0,:]*H[2,1,:]
        ),axis=1)
    constraint2 = np.stack((
        H[0,1,:]*H[0,2,:],
        H[0,1,:]*H[1,2,:] + H[1,1,:]*H[0,2,:],
        H[1,1,:]*H[1,2,:],
        H[2,1,:]*H[0,2,:] + H[0,1,:]*H[2,2,:],
        H[2,1,:]*H[1,2,:] + H[1,1,:]*H[2,2,:],
        H[2,1,:]*H[2,2,:]
        ),axis=1)
    constraint3 = np.stack((
        H[0,0,:]*H[0,2,:],
        H[0,0,:]*H[1,2,:] + H[1,0,:]*H[0,2,:],
        H[1,0,:]*H[1,2,:],
        H[2,0,:]*H[0,2,:] + H[0,0,:]*H[2,2,:],
        H[2,0,:]*H[1,2,:] + H[1,0,:]*H[2,2,:],
        H[2,0,:]*H[2,2,:]
        ),axis=1)
    tmp1 = np.stack((
        H[0,0,:]*H[0,0,:],
        H[0,0,:]*H[1,0,:] + H[1,0,:]*H[0,0,:],
        H[1,0,:]*H[1,0,:],
        H[2,0,:]*H[0,0,:] + H[0,0,:]*H[2,0,:],
        H[2,0,:]*H[1,0,:] + H[1,0,:]*H[2,0,:],
        H[2,0,:]*H[2,0,:]
        ),axis=1)
    tmp2 = np.stack((
        H[0,1,:]*H[0,1,:],
        H[0,1,:]*H[1,1,:] + H[1,1,:]*H[0,1,:],
        H[1,1,:]*H[1,1,:],
        H[2,1,:]*H[0,1,:] + H[0,1,:]*H[2,1,:],
        H[2,1,:]*H[1,1,:] + H[1,1,:]*H[2,1,:],
        H[2,1,:]*H[2,1,:]
        ),axis=1)
    tmp3 = np.stack((
        H[0,2,:]*H[0,2,:],
        H[0,2,:]*H[1,2,:] + H[1,2,:]*H[0,2,:],
        H[1,2,:]*H[1,2,:],
        H[2,2,:]*H[0,2,:] + H[0,2,:]*H[2,2,:],
        H[2,2,:]*H[1,2,:] + H[1,2,:]*H[2,2,:],
        H[2,2,:]*H[2,2,:]
        ),axis=1)
    constraint4 =  tmp1 - tmp2
    constraint5 =  tmp2 - tmp3
    constraint6 =  tmp1 - tmp3

    V[0::7,:] = constraint1
    V[1::7,:] = constraint2
    V[2::7,:] = constraint3
    V[3::7,:] = constraint4
    V[4::7,:] = constraint5
    V[5::7,:] = constraint6
    V[6::7,:] = [0,1,0,0,0,0]

    U,S,Vh = np.linalg.svd(V,full_matrices=False)
    B = Vh[-1,:]

    return B

# solve for K using alternative method
def computeK_masa(H,R,wpts,Uc=480,Vc=360,method=0):

    Hp = np.matmul(H,np.expand_dims(wpts.transpose(),axis=0))
    Rp = np.matmul(R,np.expand_dims(wpts[:,:3].transpose(),axis=0))

    Hp = Hp / np.expand_dims(Hp[:,-1,:],axis=1)
    #Hp = Hp - np.expand_dims(np.mean(Hp,axis=1),axis=1)

    # M views, N pts
    M,_,N = Rp.shape
    Rp = Rp.swapaxes(1,2).reshape((M*N,3))
    b = np.empty((M*N,1))
    b = Hp[:,0,:] + Hp[:,1,:]
    b = np.reshape(b,M*N)

    # add assumption that uc vc = image center
    # somewhat hard to justify this, but it makes the least squares solution
    # much more similar to the actual focal length
    imgCenterAssumption = Rp[:,2]*(Uc+Vc)
    b = b - imgCenterAssumption

    # method0 - f1,skew,f2
    # method1 - f,skew
    # method2 - f
    # solve intrinsic parameter using least square Ax=b
    # least square solution does not work currently...
    # or in the simplest case just take the mean over all predictions
    v1 = Rp[:,0].reshape((M*N,1))
    v2 = Rp[:,1].reshape((M*N,1))
    v3 = Rp[:,2].reshape((M*N,1))
    if method == 0:
        A = np.concatenate((v1,v2,v2),axis=1)
        m = np.linalg.lstsq(A,b)[0]
        K = [[m[0],m[1],Uc],[0,m[2],Vc],[0,0,1]]
    elif method == 1:
        A = np.concatenate((v1+v2,v2),axis=1)
        m = np.linalg.lstsq(A,b)[0]
        K = [[m[0],m[1],Uc],[0,m[0],Vc],[0,0,1]]
    elif method == 2:
        A = (v1+v2).flatten()
        f = np.mean(b/A)
        K = [[f,0,Uc],[0,f,Vc],[0,0,1]]

    return np.array(K)

'''
# compute Intrinsic Matrix from solution of SVD on all H predictions for each view
def getIntrinsics(B):

    INPUT: numpy array (6,)
    OUTPUT: numpy array (3,3)

    w = B[0]*B[2]*B[5] - B[1]*B[1]*B[5]-B[0]*B[4]*B[4] + 2*B[1]*B[3]*B[4] - B[2]*B[3]*B[3]
    d = B[0]*B[2] - B[1]*B[1]

    alpha = math.sqrt(w/(d * B[0]))
    beta = math.sqrt((w/(d*d) * B[0]))
    gamma = math.sqrt(w/(d*d * B[0])) * B[1]
    cx = (B[1]*B[4] - B[2]*B[3]) / d
    cy = (B[1]*B[3] - B[0]*B[4]) / d

    K = np.array([[alpha,gamma,cx],[0,beta,cy],[0,0,1]])

    return K
'''

def getIntrinsics(B):
    ''' alternative but equivalent formulation
    d = B[0]*B[2]-B[1]**2
    w = B[0]*B[2]*B[5]-B[1]**2*B[5]-B[0]*B[4]**2+2*B[1]*B[3]*B[4]-B[2]*B[3]**2
    fx = math.sqrt(w/(d*B[0])) if d*B[0] > 0 else math.sqrt(-1*w/(d*B[0]))
    fy = math.sqrt(w/(d**2)*B[0]) if w/B[0] > 0 else -1*math.sqrt(-1*w/(d**2)*B[0])
    skew = math.sqrt(w/(d**2*B[0]))*B[1] if w/B[0] > 0 else -1*math.sqrt(w/(d**2*B[0])*-1)
    cx = (B[1]*B[4]-B[2]*B[3])/d
    cy = (B[1]*B[3]-B[0]*B[4])/d
    '''
    cy = (B[1] * B[3] - B[0] * B[4]) / (B[0] * B[2] - B[1] * B[1])
    lmbda = B[5] - (B[3]*B[3] + cy * (B[1]*B[3] - B[0]*B[4])) / B[0]
    fx = math.sqrt(lmbda/B[0]) if lmbda/B[0] > 0 else -1*math.sqrt(lmbda/B[0]*-1)
    #print(fx)
    #print(lmbda *B[0] / (B[0]*B[2] - B[1]*B[1]))
    tmp = (lmbda*B[0]/(B[0]*B[2]-B[1]*B[1]))
    fy = math.sqrt(tmp) if tmp > 0 else -1*math.sqrt(-1*tmp)
    skew = -B[1]*fx*fx*fy / lmbda
    cx = skew*cy / fx - B[3]*fx*fx/lmbda
    K = np.array([[fx,skew,cx],[0,fy,cy],[0,0,1]])
    return K

# compute extrinsic matrix from current solution of H and K
def getExtrinsics(K,H):
    _,_,M = H.shape
    K_inv = np.linalg.inv(K)
    Rt = np.empty(H.shape)
    for i in range(M):
        scale = 1 / np.linalg.norm(K_inv.dot(np.expand_dims(H[i,:,0],axis=1)))
        #scale2 = 1 / np.linalg.norm(K_inv.dot(np.expand_dims(H[i,:,1], axis=1)))
        r1 = scale * K_inv.dot(H[i,:,0])
        r2 = scale * K_inv.dot(H[i,:,1])
        r3 = np.cross(r1,r2)
        t = scale * K_inv.dot(H[i,:,2])
        R = np.stack((r1,r2,r3),axis=0).T
        t = t.T
    return R,t

def model(params,x,y):
    K = np.zeros((3,3))
    K[0,0] += params['f1']
    K[1,1] += params['f2']
    #K[0,1] += params['skew']
    #K[0,2] += params['ux']
    #K[1,2] += params['uv']
    K[2,2] += 1
    Rt = np.zeros((3,4))
    for i in range(3):
        for j in range(3):
            Rt[i,j] += params['R' + str(int((i*3)+j))]
    for i in range(3):
        Rt[i,3] = params['t'+str(i)]

    proj = K.dot(Rt.dot(x.T))
    proj = proj / np.expand_dims(proj[-1,:],axis=0)
    error = np.linalg.norm(y.T - proj,axis=0)

    return error

def drawlmOnImage(img,lm,color=[255,0,0],size=3):
    canvas = img.copy()
    for pt in lm:
        cv2.circle(canvas,(int(pt[0]),int(pt[1])),size,color,-1)
    return canvas

################################################################################################
# main methods depending on mode
def face_module():
    data = sio.loadmat(args.matfile)
    lm3d = data['lm3d']

    uvcoord,worldcoord = getCoordinates(lm3d)
    N,d,M = worldcoord.shape

    # visualize scatter plot of 3d landmarks on the projected plane
    #plt.scatter(worldcoord[:,0,:].flatten(),worldcoord[:,1,:].flatten())
    #plt.show()
    #print(worldcoord.shape)
    #quit()

    # get initial estimates on camera params
    H,K,Rt = calibrate(uvcoord,worldcoord)
    print(K)

    '''
    # optimize initial guess on H
    H_opt = np.empty(H.shape)
    for i in range(M):
        xdata = worldcoord[:,:,i]
        ydata = uvcoord[:,:,i].flatten()

        popt,pcov = curve_fit(model,xdata,ydata)
        H_opt[:,:,i] = popt.reshape((3,3)).T

        #error_preopt = np.linalg.norm(uvcoord[:,:2,i] - xdata.dot(H[:,:,i])[:,:2],axis=1)
        #error_postopt = np.linalg.norm(uvcoord[:,:2,i] - xdata.dot(popt.reshape((3,3)))[:,:2],axis=1)
        #print(f"pre optimized: {np.mean(error_preopt)} | post optimized: {np.mean(error_postopt)}")

    # get new intrinsic extrinsic params
    Knew,Rtnew = getParams(H_opt)
    tmp = K.dot(Rt[:,:,0].dot(worldcoord[:,:,0].T)).T
    tmp = tmp / np.expand_dims(tmp[:,-1],axis=1)

    tmp2 = Knew.dot(Rtnew[:,:,0].dot(worldcoord[:,:,0].T)).T
    tmp2 = tmp2 / np.expand_dims(tmp2[:,-1],axis=1)
    '''

    tmp = K.dot(Rt[:,:,0].dot(worldcoord[:,:,0].T)).T
    tmp = tmp / np.expand_dims(tmp[:,-1],axis=1)
    print(f"2D error: {np.mean(np.linalg.norm(uvcoord[:,:,0] - tmp,axis=1))}")
    #print(f"error: {np.mean(np.linalg.norm(uvcoord[:,:,0] - tmp2,axis=1))}")

# solve the checkerboard 3d object using direct linear transform instead of homography
# imgpts dim is Nx3xM
# worldpts dim is Nx4xM
def face_module2():
    data = sio.loadmat(args.matfile)
    lm3d = data['lm3d']
    imgs = data['imgs']

    uvpts = lm3d[:,:2,:]
    worldpts = sio.loadmat('../matlab/facelm3d.mat')['meu3d']
    worldpts = worldpts * -1

    N,_,M = uvpts.shape
    uvcoord = np.ones((N,3,M))
    worldcoord = np.ones((N,4,M))

    uvcoord[:,:2,:] = uvpts
    worldcoord[:,:3,:] = np.expand_dims(worldpts,axis=2)

    # get initial estimate on camera params
    H,K,Rt = calibrate2(uvcoord,worldcoord)

    # analyze initial guess reprojection error
    errors = []
    for i in range(M):
        print(H[:,:,i].dot(worldcoord[:,:,i].T).T)
        print(H[:,:3,i])
        print(H[:,:3,i].dot(worldcoord[:,:3,i].T).T)
        tmp = H[:,:,i].dot(worldcoord[:,:,i].T).T
        reprojection = tmp/np.expand_dims(tmp[:,-1],axis=-1)
        error = np.mean(np.linalg.norm(reprojection - uvcoord[:,:,i],axis=1))
        errors.append(error)

        # if you want to save the image
        img = drawlmOnImage(imgs[i,:,:,:],reprojection)
        img = drawlmOnImage(img,uvpts[:,:,i],color=[0,255,0],size=5)
        cv2.imshow('reprojection',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        #plt.imsave(os.path.join('tmp','img' + "%04d"%i + '.png'),img)
    print(f"mean 2d reprojection error: {np.mean(errors)}")

def face_module___(uvcoord,worldcoord):
    # get initial estimate on camera Params
    # compute rotation for each view of the points as well using weak perspective projection model

    M = uvcoord.shape[-1]
    N = uvcoord.shape[0]
    H = np.empty((M,3,4))
    errors = []
    worldcoord = worldcoord - np.min(worldcoord,axis=0)
    wpts, normMatrix3d = normalize3d(worldcoord)
    valid_pts = np.all(uvcoord > -1, axis=1)
    #normMatrix2d = util.normalize2d(uvcoord)
    #norm2d_inv = np.linalg.inv(normMatrix2d)

    #canvas = util.drawpts(canvas, tmp[:,:2])
    #cv2.imshow('canvas', canvas.astype(np.uint8))
    #cv2.waitKey(0)
    #quit()

    for i in range(M):
        mask = valid_pts[:,i]
        uv = uvcoord[mask,:,i]
        uv, normMatrix2d = normalize2d(uv)
        norm2d_inv = np.linalg.inv(normMatrix2d)
        #uv = normMatrix2d.dot(uv.T).T
        xyz = wpts[mask]

        # wpt_test = normMatrix3d.dot(worldcoord.T).T
        # s,R[i,:,:],t[i,:] = computeWeakProjection(uv[:,:2],wpts[:,:3])
        P,error = getDLT(uv,xyz)

        # reverse normalization on P
        # http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
        H[i,:,:] = norm2d_inv.dot(P.dot(normMatrix3d))

        # restrict answers to positive depth
        if np.sum(H[i,-1,:]) < 0:
            H[i,:,:] *= -1

        # get error
        tmp = H[i,:,:].dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)

        canvas = np.zeros((500,500,3))
        canvas = util.drawpts(canvas,reprojection)
        canvas = util.drawpts(canvas,uvcoord[mask,:2,i],color=[255,0,0])
        cv2.imshow('canvas', canvas.astype(np.uint8))
        cv2.waitKey(0)

        error = np.mean(np.linalg.norm(reprojection[mask,:2] - uvcoord[mask,:2,i],axis=1))
        errors.append(error)

    quit()
    print(f"mean reprojection error: {np.mean(errors)}")
    #quit()
    #plotHistogram(np.array(errors))
    #plotHistogram(np.array(errors),save=True, filename='tmp/reprojection_error2D.png')
    #plt.hist(errors,bins=100)
    #plt.show()

def face_module3(uvcoord,worldcoord):
    # get initial estimate on camera Params
    # compute rotation for each view of the points as well using weak perspective projection model

    M = uvcoord.shape[-1]
    N = uvcoord.shape[0]
    H = np.empty((M,3,4))
    errors = []
    n_wpts, normMatrix3d = normalize3d(worldcoord)
    n_uvpts, normMatrix2d = util.normalize2d(uvcoord)
    norm2d_inv = np.linalg.inv(normMatrix2d)
    valid_pts = np.all(uvcoord > -1, axis=1)

    #canvas = np.zeros((500,500,3))
    #canvas = util.drawpts(canvas,worldcoord[:,:2])
    #cv2.imshow('world pts', canvas.astype(np.uint8))
    #cv2.waitKey(0)

    for i in range(M):
        mask = valid_pts[:,i]
        uv = n_uvpts[i].T
        xyz = n_wpts[mask]

        # wpt_test = normMatrix3d.dot(worldcoord.T).T
        # s,R[i,:,:],t[i,:] = computeWeakProjection(uv[:,:2],wpts[:,:3])
        P = getDLT(uv,xyz)

        # reverse normalization on P
        # http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
        H[i,:,:] = norm2d_inv.dot(P.dot(normMatrix3d))

        # restrict answers to positive depth
        if np.sum(H[i,-1,:]) < 0:
            H[i,:,:] *= -1
        tmp = H[i,:,:].dot(worldcoord.T).T

        # get error
        tmp = H[i,:,:].dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
        #canvas = np.zeros((500,500,3))
        #canvas = util.drawpts(canvas,reprojection[mask])
        #canvas = util.drawpts(canvas,uvcoord[:,:2,i],color=[255,0,0])
        #cv2.imshow('canvas', canvas.astype(np.uint8))
        #cv2.waitKey(1)
        error = np.mean(np.linalg.norm(reprojection[mask,:2] - uvcoord[mask,:2,i],axis=1))
        errors.append(error)

    print(f"mean reprojection error: {np.mean(errors)}")
    #quit()
    #plotHistogram(np.array(errors))
    #plotHistogram(np.array(errors),save=True, filename='tmp/reprojection_error2D.png')
    #plt.hist(errors,bins=100)
    #plt.show()

    # We get one estimation of the Intrinsic Matrix per view
    K = np.empty((M,3,3))
    Rt = np.empty((M,3,4))
    Pc = np.empty((M,3,1))
    Pv = np.empty((M,3))
    for i in range(M):
        # P = [K*R_ | -K*R_*Pc]
        K_,R_,Pc_,pp_,pv_ = util.decompose_projection_matrix(H[i,:,:])

        s = K_[2,2]
        K[i] = K_/s
        Rt[i,:3,:3] = R_

        #M = H[i,:3,:3]
        #Rt_tmp = np.concatenate((R_,-R_.dot(Pc_)), axis=-1)
        #w = Rt_tmp.dot(worldcoord.T)

        '''
        A = np.zeros((68,2))
        A[:,0] = w[0,:]
        A[:,1] = w[2,:]
        b = w[-1,:] * uvcoord[:,0,i]
        x = np.linalg.lstsq(A,b)
        f1,ux = x[0]
        #print(x)

        A = np.zeros((68,2))
        A[:,0] = w[1,:]
        A[:,1] = w[2,:]
        b = w[-1,:] * uvcoord[:,1,i]
        x = np.linalg.lstsq(A,b)
        f2,uy = x[0]
        #print(x)

        K_tmp = np.array([[f1,0,ux],[0,f2,uy],[0,0,-1]])
        print(K_/s)
        print(K_tmp)

        P = np.concatenate((K_.dot(R_),-K_.dot(R_.dot(Pc_))), axis=-1)
        tmp = P.dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
        error = np.mean(np.linalg.norm(reprojection[:,:2] - uvcoord[:,:2,i],axis=1))
        print(error)

        K_tmp = np.array([[f1,0,ux],[0,f2,uy],[0,0,-1]])
        P = np.concatenate((K_tmp.dot(R_),-K_tmp.dot(R_.dot(Pc_))), axis=-1)
        tmp = P.dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)

        print(K_tmp)
        print(error)

        A = np.zeros((2*68,3))
        A[:68,0] = w[0,:]
        A[68:,1] = w[1,:]
        A[68:,2] = w[2,:]
        A[:68,2] = w[2,:]
        b1 = w[-1,:] * uvcoord[:,0,i]
        b2 = w[-1,:] * uvcoord[:,1,i]
        b = np.concatenate((b1,b2))
        x = np.linalg.lstsq(A,b)
        print(x)
        quit()

        #print(R_)
        #R_[[0,2]][:,[0,2]]
        #b = np.linalg.solve(R_.T,M[0,:])
        #b2 = np.linalg.solve(R_[[0,2]][:,[0,2]].T,M[0,[0,2]])
        #b3 = np.linalg.lstsq(R_.T[:,[0,2]],M[0,:])[0]

        A = np.zeros((6,3))
        A[:3,0] = R_[0,:]
        A[:3,1] = R_[2,:]
        A[3:6,0] = R_[1,:]
        A[3:6,2] = R_[2,:]
        b = M[:2,:].flatten()
        x = np.linalg.lstsq(A,b)[0]
        K_tmp = np.array([[x[0],0,x[1]],[0,x[0],x[2]],[0,0,s]])

        P = np.concatenate((K_.dot(R_),-K_.dot(R_.dot(Pc_))), axis=-1)
        print(P)
        tmp = P.dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
        error = np.mean(np.linalg.norm(reprojection[:,:2] - uvcoord[:,:2,i],axis=1))
        print(error)

        K_tmp = np.array([[x[0],0,x[1]],[0,x[0],x[2]],[0,0,s]])
        P = np.concatenate((K_tmp.dot(R_),-K_tmp.dot(R_.dot(Pc_))), axis=-1)
        print(P)
        tmp = P.dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
        quit()

        print(K_/s)
        print(K_tmp/s)
        quit()

        w = R_.dot(worldcoord[:,:3].T)
        y = M.dot(worldcoord[:,:3].T)

        A = np.zeros((2*68,4))
        A[0::2,0] = w[0,:]
        A[0::2,2] = w[2,:]
        #A[1::2,1] = w[1,:]
        #A[1::2,3] = w[2,:]

        b = np.zeros((2*68))
        b[0::2] = y[0,:]
        b[1::2] = y[1,:]

        x = np.linalg.lstsq(A,b)[0]

        print(A[0::2])
        print(b[0::2])
        print(x)
        quit()

        P = np.concatenate((K_.dot(R_),-K_.dot(R_.dot(Pc_))), axis=-1)
        print(P)
        tmp = P.dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
        print(reprojection)
        error = np.mean(np.linalg.norm(reprojection[:,:2] - uvcoord[:,:2,i],axis=1))
        print(error)

        K_tmp = np.array([[x[0],0,x[2]],[0,x[1],x[3]],[0,0,s]])
        P = np.concatenate((K_tmp.dot(R_),-K_tmp.dot(R_.dot(Pc_))), axis=-1)
        print(P)
        tmp = P.dot(worldcoord.T).T
        reprojection = tmp / np.expand_dims(tmp[:,-1],axis=-1)
        print(reprojection)
        quit()
        error = np.mean(np.linalg.norm(reprojection[:,:2] - uvcoord[:,:2,i],axis=1))
        print(error)

        print(K_/s)
        quit()

        print(M)
        print(b/s)
        print(b2/s)
        print(b3/s)
        print(np.expand_dims(b,0).dot(R_))
        print(np.expand_dims(np.array([b2[0],0,b2[1]]),0).dot(R_))
        print(np.expand_dims(np.array([b3[0],0,b3[1]]),0).dot(R_))
        quit()

        print(K_)
        print(K[i])
        '''

        Pc[i] = Pc_
        Pv[i] = pv_
        Rt[i,:,3] = np.squeeze(-R_.dot(Pc_))

        #tmp = Rt[i].dot(worldcoord.T).T
        #print(tmp[:,-1] > 0)
        #n2d_inv = np.linalg.inv(n_transform2d[i])
        #K[i] = n2d_inv.dot(K_)/s

    return K, Rt, errors

    w = np.matmul(Rt,np.tile(worldcoord.T,(M,1,1)))
    w = np.swapaxes(w,1,2).reshape((N*M,3))
    uv = uvcoord.reshape((N*M,3))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(w[:,0],w[:,1],w[:,2])
    ax.scatter(0,0,0)
    plt.show()
    plt.close()
    quit()

    #np.set_printoptions(threshold=np.inf)
    b1 = uv[:,0] * w[:,-1]
    b2 = uv[:,1] * w[:,-1]

    A = np.zeros((N*M,3))
    A[:,0] = w[:,0]
    A[:,1] = w[:,2]
    A[:,2] = 0
    B = np.zeros((N*M,3))
    B[:,0] = w[:,1]
    B[:,1] = 0
    B[:,2] = w[:,2]

    xyz = pptk.rand(100,3)
    v = pptk.viewer(xyz)

    quit()
    np.concatenate((A,B),axis=0)
    np.concatenate((b1,b2),axis=0)

    A = np.zeros((N*M,2))
    A[:,0] = w[:,0]
    A[:,1] = w[:,2]
    x = np.linalg.lstsq(A,b1)
    f1,ux = x[0]

    A = np.zeros((N*M,2))
    A[:,0] = w[:,1]
    A[:,1] = w[:,2]
    x = np.linalg.lstsq(A,b2)
    f2,uy = x[0]

    K_tmp = np.array([[f1,0,ux],[0,f2,uy],[0,0,1]])
    print(K_tmp)
    quit()

    print(np.mean(K,axis=0))
    A = np.zeros((N*M,2))
    K_mean = np.mean(K,axis=0)
    print(K_mean)

    quit()

    '''
    for i in range(M):
        # proj = K[i].dot(Rt[i].dot(worldcoord.T))
        # proj = proj / np.expand_dims(proj[-1,:],axis=0)
        # error = np.mean(np.linalg.norm(uvcoord[:,:,i].T - proj,axis=0))
        print(i)
        print(K[i])
        K_opt, Rt_opt = refineParams(K[i],Rt[i],worldcoord,uvcoord[:,:,i])
        print(K_opt)
        K[i] = K_opt
        Rt[i] = Rt_opt
    '''

    #K_opt, Rt_opt = refineParams(K_mean,Rt,worldcoord,uvcoord)
    # K = np.absolute(K)
    #plotHistogram(K[:,0,0],title='F1')
    #plotHistogram(K[:,1,1],title='F2')
    #plotHistogram(K[:,0,2],title='Uc')
    #plotHistogram(K[:,1,2],title='Vc')

    return K, Rt, errors

# generic calibration using corresponding 2d 3d points
# uvcoordinates and world coordinates are in homogeneous coordinate system
# uvcoord: (M,8,3)
# worldcoord: (M,8,4)
def calibrate(uvcoord,worldcoord):
    # get initial estimate on camera Params
    # compute rotation for each view of the points as well using weak perspective projection model

    M = uvcoord.shape[0]
    H = np.empty((M, 3, 4))
    errors = []
    for i in range(M):
        uv, normMatrix2d = normalize2d(uvcoord[i, :, :])
        wpts, normMatrix3d = normalize3d(worldcoord[i,:,:])

        P, error = getDLT(uv, wpts)
        errors.append(error)

        # reverse normalization on P
        # http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
        norm2d_inv = np.linalg.inv(normMatrix2d)
        H[i, :, :] = norm2d_inv.dot(P.dot(normMatrix3d))

    plotHistogram(np.array(errors))

    # We get one estimation of the Intrinsic Matrix per view
    K = np.empty((M, 3, 3))
    Rt = np.empty((M, 3, 4))
    Pc = np.empty((M, 3, 1))
    Pv = np.empty((M, 3))
    for i in range(M):
        # P = [K*R_ | -K*R_*Pc]
        K_, R_, Pc_, pp_, pv_ = util.decompose_projection_matrix(H[i, :, :])
        s = K_[2, 2]
        K[i] = K_ / s
        Rt[i, :3, :3] = R_
        Pc[i] = Pc_
        Pv[i] = pv_
        Rt[i, :, 3] = np.squeeze(-R_.dot(Pc_))

    return K, Rt

# face module 4 which uses multiple objects
def face_module4(uvcoord,worldcoord):
    # get initial estimate on camera Params
    # compute rotation for each view of the points as well using weak perspective projection model
    M = uvcoord.shape[-1]
    N = uvcoord.shape[0]
    H = np.empty((M,3,4))

    # normalize 2d point set such that x' = Px
    n_uvpts, normMatrix2d = util.normalize2d(uvcoord)
    norm2d_inv = np.linalg.inv(normMatrix2d)
    valid_pts = np.all(uvcoord > -1, axis=1)

    # normalize 3d point set such that x' = Px
    n_wpts, normMatrix3d = util.normalize3d(worldcoord)

    scales = []
    errors = []
    for i in range(M):

        uv = n_uvpts[i].T
        wpts = n_wpts[i].T
        P = getDLT(uv,wpts)

        # reverse normalization on P
        # http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
        norm2d_inv = np.linalg.inv(normMatrix2d)
        P = norm2d_inv.dot(P.dot(normMatrix3d))
        H[i] = P

        # reprojection onto image plane to get error
        tmp = H[i].dot(worldcoord[i].T).T
        reprojection = tmp/np.expand_dims(tmp[:,-1],axis=-1)

        scales.append(tmp[:,-1])
        error = np.mean(np.linalg.norm(reprojection - uvcoord[:,:,i],axis=1))
        errors.append(error)

    print(f"mean reprojection error: {np.mean(errors)}")
    #quit()
    #plotHistogram(np.array(errors))
    #plotHistogram(np.array(errors))
    #hist = np.array(scales).flatten()
    #plt.hist(hist,bins=100)
    #plt.savefig('face_module3-scaledistribution.png')
    #plt.close()

    # We get one estimation of the Intrinsic Matrix per view
    K = np.empty((M,3,3))
    Rt = np.empty((M,3,4))
    Pc = np.empty((M,3,1))
    Pv = np.empty((M,3))
    for i in range(M):
        # P = [K*R_ | -K*R_*Pc]
        K_,R_,Pc_,pp_,pv_ = util.decompose_projection_matrix(H[i])
        s = K_[2,2]
        K[i] = K_/s
        Rt[i,:3,:3] = R_
        Pc[i] = Pc_
        Pv[i] = pv_
        Rt[i,:,3] = np.squeeze(-R_.dot(Pc_))

    # optimize initial guess on H
    #for i in range(M):
        # proj = K[i].dot(Rt[i].dot(worldcoord.T))
        # proj = proj / np.expand_dims(proj[-1,:],axis=0)
        # error = np.mean(np.linalg.norm(uvcoord[:,:,i].T - proj,axis=0))
    #    K_opt, Rt_opt = refineParams(K[i],Rt[i],worldcoord[:,:,i],uvcoord[:,:,i])
    #    K[i] = K_opt
    #    Rt[i] = Rt_opt

    return K, Rt, errors

def checkerboard_module2():
    imgpts = sio.loadmat('../matlab/imgpts.mat')['imagePoints']
    worldpts = sio.loadmat('../matlab/worldpts.mat')['worldPoints']

    N,_,M = imgpts.shape
    uvcoord = np.ones((N,3,M))
    worldcoord = np.ones((N,4,M))

    uvcoord[:,:2,:] = imgpts
    worldcoord[:,:2,:] = np.expand_dims(worldpts,axis=2)

    # get initial estimate on camera params
    H,K,Rt = calibrate2(uvcoord,worldcoord)

def checkerboard_module3():
    imgpts = sio.loadmat('../matlab/imgpts.mat')['imagePoints']
    worldpts = sio.loadmat('../matlab/worldpts.mat')['worldPoints']

    N,_,M = imgpts.shape
    uvcoord = np.ones((N,3,M))
    worldcoord = np.ones((N,3,M))

    # world coordinates are actually always the same buut just in case they are not...
    uvcoord[:,:2,:] = imgpts
    worldcoord[:,:2,:] = np.expand_dims(worldpts,axis=2)

    # get initial estimate on camera Params
    # compute rotation for each view of the points as well
    H = np.empty((M,3,3))
    R = np.empty((M,3,3))
    t = np.empty((M,3))
    for i in range(M):
        s,R[i,:,:],t[i,:] = computeWeakProjection(uvcoord[:,:2,i],worldcoord[:,:,i])
        H[i,:,:],_ = getHomography(uvcoord[:,:,i],worldcoord[:,:,i])

    K = computeK_masa(H,R,worldcoord[:,:3,0],Uc=480,Vc=360,method=0)

    #B = computeB(H)
    #K = getIntrinsics(B)
    #R,T = getExtrinsics(K,H)
    #Rt = np.concatenate((R,np.expand_dims(T,1)),axis=1)

# solve the camera calibration using Zhang's planar object method
# imgpts is Nx3xM
# worldpts is Nx3xM
def checkerboard_module():

    imgpts = sio.loadmat('../matlab/imgpts.mat')['imagePoints']
    worldpts = sio.loadmat('../matlab/worldpts.mat')['worldPoints']

    N,_,M = imgpts.shape
    uvcoord = np.ones((N,3,M))
    worldcoord = np.ones((N,3,M))

    uvcoord[:,:2,:] = imgpts
    worldcoord[:,:2,:] = np.expand_dims(worldpts,axis=2)

    # get initial estimate on camera Params
    H,K,Rt = calibrate(uvcoord,worldcoord)

    print(K)
    '''
    # optimize initial guess on H
    H_opt = H.copy()
    for i in range(M):
        xdata = worldcoord[:,:,i]
        ydata = uvcoord[:,:,i].flatten()

        popt,pcov = curve_fit(model,xdata,ydata,H_opt[:,:,i].T)
        H_opt[:,:,i] = popt.reshape((3,3)).T
        #print(np.linalg.norm(ydata.reshape((56,3)) - xdata.dot(popt.reshape((3,3))),axis=1))
        #quit()

    #error_preopt = np.linalg.norm(uvcoord[:,:,i] - xdata.dot(H[:,:,i])[:,:],axis=1)
    #error_postopt = np.linalg.norm(uvcoord[:,:,i] - xdata.dot(popt.reshape((3,3)))[:,:],axis=1)
    #print(f"pre optimized: {np.mean(error_preopt)} | post optimized: {np.mean(error_postopt)}")

    # get new intrinsic extrinsic params
    Knew,Rtnew = getParams(H_opt)
    tmp = K.dot(Rt[:,:,0].dot(worldcoord[:,:,0].T)).T
    tmp = tmp / np.expand_dims(tmp[:,-1],axis=1)

    tmp2 = Knew.dot(Rtnew[:,:,0].dot(worldcoord[:,:,0].T)).T
    tmp2 = tmp2 / np.expand_dims(tmp2[:,-1],axis=1)

    print(Knew)
    '''

    tmp = K.dot(Rt[:,:,0].dot(worldcoord[:,:,0].T)).T
    tmp = tmp / np.expand_dims(tmp[:,-1],axis=1)
    print(f"2D error: {np.mean(np.linalg.norm(uvcoord[:,:,0] - tmp,axis=1))}")
    #print(f"error: {np.mean(np.linalg.norm(uvcoord[:,:,0] - tmp2,axis=1))}")

# more realistic dataset using 3d landmark locations with corresponding 2d locations
# found via face-alignment method
def getFaceDataBIWI(subject,maxangle=20, lmrange=(0,68)):
    lm2d_path = f"/home/huynshen/projects/git/landmark-detection/SAN/tmp/biwi_kinect_2dlm/{subject}/lm2d.mat"
    lm3d_path = f"../matlab/sub{subject}_meu3d.mat"

    # load the data
    # data[lm2d] provides confidence values along with each landmark point
    data2d = sio.loadmat(lm2d_path)
    lm2d = data2d['lm2d']
    #lm2d[:,:,[0,1]] = lm2d[:,:,[1,0]]
    data3d = sio.loadmat(lm3d_path)
    worldpts = data3d['meu3d']
    #worldpts[:,[0,1]] = worldpts[:,[1,0]]

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
        R, center = util.read_pose(p)
        rx,ry,rz = [util.convertRadiansToDegrees(theta) for theta in util.getEulerAngles(R)]
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
    #worldcoord[:,[0,1]] = worldcoord[:,[1,0]]
    #worldcoord[:,0] *= -1

    uvcoord = np.swapaxes(uvcoord,0,2)
    uvcoord = np.swapaxes(uvcoord,0,1)
    #uvcoord[:,:,[0,1]] = uvcoord[:,:,[1,0]]
    #uvcoord[:,:,:] *= -1

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

def getFaceDataMasa2():
    data = sio.loadmat('/home/huynshen/projects/git/landmark-detection/SAN/tmp/masa_only3/lm2d.mat')
    lm2d = data['lm2d']
    imgs = data['files']

    #worldpts = sio.loadmat('../matlab/facelm3d.mat')['meu3d']
    worldpts = sio.loadmat('/home/huynshen/projects/face-camera-calibration/facedetector/tmp/masa_only3/lm3d.mat')['lm3d']
    worldpts = np.mean(worldpts,axis=-1)
    worldpts = worldpts - np.expand_dims(np.mean(worldpts,axis=0),axis=0)
    worldpts = worldpts - np.min(worldpts,axis=0)
    M, N, _  = lm2d.shape

    # get rough idea of depth using checkerboard info
    data = sio.loadmat('/home/huynshen/projects/face-camera-calibration/matlab/masa_only3_calib.mat')
    t = data['translations']
    rots = data['rotations']
    pts = data['worldPoints']
    files = data['imageFileNames']
    checkerboard = np.zeros((63, 4))
    checkerboard[:,-1] = 1
    checkerboard[:,:2] = np.expand_dims(pts,axis=0)
    Rt = np.concatenate((rots,np.expand_dims(t.T,axis=1)),axis=1)
    Rt = np.swapaxes(Rt,0,-1)
    Rt = np.swapaxes(Rt,1,2)
    proj = np.matmul(Rt,np.expand_dims(checkerboard.T,axis=0))
    depth = np.linalg.norm(proj,axis=1)
    mean_depth = np.mean(depth,axis=1)
    gt_files = []
    for f in files[0]:
        gt_files.append(f[0])

    # 122 is the average intraocular distance for a male individual
    ratio = 122 / np.linalg.norm(worldpts[36, :] - worldpts[46, :])
    worldpts = worldpts * ratio

    # copy data to formatted output
    uvcoord = np.ones((M, N, 3))
    worldcoord = np.ones((N, 4))
    uvcoord[:, :, :2] = lm2d[:,:,:2]
    worldcoord[:, :3] = worldpts
    uvcoord = np.swapaxes(uvcoord,0,2)
    uvcoord = np.swapaxes(uvcoord,0,1)

    return uvcoord, worldcoord, imgs, mean_depth, np.array(gt_files)

def getFaceDataMasa():
    data = sio.loadmat('/home/huynshen/projects/git/landmark-detection/SAN/tmp/masa_only/lm2d.mat')
    lm2d = data['lm2d']
    imgs = data['files']

    worldpts = sio.loadmat('../matlab/facelm3d.mat')['meu3d']
    M, N, _  = lm2d.shape

    # 122 is the average intraocular distance for a male individual
    ratio = 122 / np.linalg.norm(worldpts[36, :] - worldpts[46, :])
    worldpts = worldpts * ratio

    # copy data to formatted output
    uvcoord = np.ones((M, N, 3))
    worldcoord = np.ones((N, 4))
    uvcoord[:, :, :2] = lm2d[:,:,:2]
    worldcoord[:, :3] = worldpts
    uvcoord = np.swapaxes(uvcoord,0,2)
    uvcoord = np.swapaxes(uvcoord,0,1)

    return uvcoord, worldcoord, imgs

# Synthetic face dataset where 3d landmark locations correspond to
# 2D images using known calibration and extrinsic parameters from a checkerboard
def getSynthFaceData():
    data = sio.loadmat('../matlab/synth_uvface.mat')
    #print(data['synthdata']['K'][0,0])
    #quit()

    imgs = sio.loadmat(args.matfile)['imgs']
    uvpts = data['synthdata']['uvpts'][0,0]
    # uvpts[:,2::-1] = uvpts[:,0:2]

    worldpts = sio.loadmat('../matlab/facelm3d.mat')['meu3d']
    #worldpts[:,1] = worldpts[:,1]*-1

    #122 is the average intraocular distance for a male individual
    ratio = 122 / np.linalg.norm(worldpts[36,:] - worldpts[46,:])
    worldpts = worldpts*ratio

    N, _, M = uvpts.shape
    uvcoord = np.ones((N,3,M))
    worldcoord = np.ones((N,4))

    uvcoord[:,:2,:] = uvpts
    worldcoord[:,:3] = worldpts

    return uvcoord,worldcoord, imgs

# Synthetic face dataset where 3d landmark locations correspond to
# 2D images using Weak Perspective transformation
def getSynthFaceDataWeak():
    data = sio.loadmat('../matlab/synth_uvface_weak.mat')
    imgs = sio.loadmat(args.matfile)['imgs']
    uvpts = data['synthdata']['uvpts'][0,0]
    # uvpts[:,2::-1] = uvpts[:,0:2]

    worldpts = sio.loadmat('../matlab/facelm3d.mat')['meu3d']
    #worldpts[:,1] = worldpts[:,1]*-1

    #122 is the average intraocular distance for a male individual
    ratio = 122 / np.linalg.norm(worldpts[36,:] - worldpts[46,:])
    worldpts = worldpts*ratio

    N, _, M = uvpts.shape
    uvcoord = np.ones((N,3,M))
    worldcoord = np.ones((N,4))

    uvcoord[:,:2,:] = uvpts
    worldcoord[:,:3] = worldpts

    return uvcoord,worldcoord, imgs

# normalize 2d points
def normalize2d(uvpts):
    mean_translation = np.expand_dims(np.mean(uvpts,axis=0),axis=0)
    uvpts = uvpts - mean_translation
    scale = np.sqrt(2) / np.mean(np.linalg.norm(uvpts[:,:2],axis=1))
    uvpts = uvpts * scale
    uvpts[:,2] = 1

    nmatrix = np.zeros((3,3))
    nmatrix[:,2] = -scale*mean_translation
    nmatrix[0,0] = scale
    nmatrix[1,1] = scale
    nmatrix[2,2] = 1

    return uvpts, nmatrix

# normalize 2d points
def normalize3d(pts):
    mean_translation = np.expand_dims(np.mean(pts,axis=0),axis=0)
    pts = pts - mean_translation
    scale = np.sqrt(3) / np.mean(np.linalg.norm(pts[:,:3],axis=1))
    pts = pts*scale
    pts[:,3] = 1

    nmatrix = np.zeros((4,4))
    nmatrix[:,3] = -scale*mean_translation
    nmatrix[0,0] = scale
    nmatrix[1,1] = scale
    nmatrix[2,2] = scale
    nmatrix[3,3] = 1

    return pts, nmatrix

# Refine intrinsic and extrinsic matrix using Lavenberg-Marquardt optimization for local minima
# This is the typical strategy to optimize the initial closed form solution
def refineParams(K,Rt,x,y):
    params = Parameters()
    #params.add('f1', value=K[0,0],min=0,max=2000)
    #params.add('f2', value=K[1,1],min=0,max=2000)
    #params.add('skew',value=K[0,1],min=-0.001,max=0.001)
    #params.add('ux', value=K[0,2],min=-960,max=960)
    #params.add('uv', value=K[1,2],min=-720,max=720)
    params.add('f1', value=K[0, 0])
    params.add('f2', value=K[1, 1])
    #params.add('skew', value=K[0, 1])
    #params.add('ux', value=K[0, 2])
    #params.add('uv', value=K[1, 2])
    for c, val in enumerate(Rt[:3,:3].flatten()):
        params.add('R'+str(c), value=val)
    for c, val in enumerate(Rt[:,3].flatten()):
        params.add('t'+str(c),value=val)

    # proj = K.dot(Rt.dot(x.T))
    # proj = proj / np.expand_dims(proj[-1, :], axis=0)
    # error = np.mean(np.linalg.norm(y.T - proj, axis=0))

    opt = Minimizer(model,params,fcn_args=(x,y))
    result = opt.minimize()
    K = np.zeros((3,3))
    p = result.params
    K[0,0] = p['f1']
    K[1,1] = p['f2']
    #K[0,1] = p['skew']
    #K[0,2] = p['ux']
    #K[1,2] = p['uv']
    K[0,2] = 320
    K[1,2] = 240
    K[2,2] = 1
    for i in range(3):
        for j in range(3):
            Rt[i,j] = p['R'+str(int(i*3+j))]
    for i in range(3):
        Rt[i,3] = p['t'+str(i)]

    #report_fit(result)
    return K,Rt

# this method finds a proper rotation matrix given a prediction of one
def refineR(R):
    return

# plot line graph
def plotLine(values):

    return

# custom function to plot histogram
# https://matplotlib.org/3.1.1/gallery/statistics/histogram_features.html
def plotHistogram(values,bincount=50,save=False,filename='output.png',title='plot'):
    fig, ax= plt.subplots()
    mu = np.mean(values)
    sigma = np.std(values)
    if len(values) < 100:
        bincount = len(values)
    n, bins, patches = ax.hist(values, bins=bincount, histtype='bar')

    #n, bins, patches = ax.distplot(values, bins=bincount)
    # best fit curve gaussian
    #y = ((1 / np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)
    #ax.plot(bins, y, '--')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'{title}: $\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
    plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
    if not save:
        plt.show()
    else:
        plt.savefig(filename)

def save2DReprojection(K,Rt,uvpts,worldpts,imgs,filename='tmp'):
    M = K.shape[0]
    valid_pts = np.all(uvpts > -1,axis=1)
    for i in range(M):
        if type(imgs[i]) == np.str_:
            canvas = imageio.imread(imgs[i])
        else:
            canvas = imgs[i]
        mask = valid_pts[:,i]

        H = K[i].dot(Rt[i])
        tmp = H.dot(worldpts.T).T
        reprojection = tmp/np.expand_dims(tmp[:, -1], axis=-1)
        #error = np.mean(np.linalg.norm(reprojection - uvpts[:, :, i], axis=1))

        img = drawlmOnImage(canvas, reprojection[mask],size=5)
        img = drawlmOnImage(img, uvpts[mask, :, i], color=[0, 255, 0])
        cv2.imshow('reprojection', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        plt.imsave(os.path.join(filename,'img' + "%04d"%i + '.png'),img)

# view the reprojection of the world points to camera centric coordinates
def saveCameraCentricProjection(K,Rt,worldpts):
    return

#convert to weak perspective projection
def convertWorldToWeak2D(uvpts, worldpts):
    M = uvpts.shape[-1]
    weak_uvpts = np.empty(uvpts.shape)
    worldpts[:,1] *= -1
    for i in range(M):
        s, R, T = computeWeakProjection(uvpts[:,:2,i],worldpts[:,:3])
        proj = s*(R.dot(worldpts[:,:3].T) + np.expand_dims(T,axis=1))
        proj = proj / np.expand_dims(proj[-1],axis=0)
        print(proj[:2,:].T)
        quit()
        error = np.linalg.norm(uvpts[:,:2,i] - proj[:2,:].T,axis=1)
        print(error)
        #print(proj)
        #quit()
        #print(proj / np.expand_dims(proj[-1,:],axis=0))
        quit()
        weak_uvpts[:,:,i] = proj.T

    return weak_uvpts
####################################################################################################
####################################################################################################
####################################################################################################
# main execution entry point
if __name__ == '__main__':
    if args.mode == 'face':
        face_module()
    elif args.mode == 'face2':
        uvpts, worldpts = getFaceData()
        face_module3(uvpts, worldpts)
    elif args.mode == 'face_synth_weak':
        uvpts, worldpts, imgs = getSynthFaceDataWeak()
        imgs *= 0
        K, Rt = face_module3(uvpts, worldpts)
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        plotHistogram(K[:, 0, 0], save=True, filename='tmp/f1.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        save2DReprojection(K, Rt, uvpts, worldpts, imgs,filename='tmp')
    elif args.mode == 'face3':
        uvpts, worldpts, imgs = getSynthFaceData()
        imgs *= 0
        K, Rt = face_module3(uvpts, worldpts)
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        plotHistogram(K[:, 0, 0], save=True, filename='tmp/f1.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename='tmp')

    elif args.mode == 'face_masa':
        uvpts, worldpts, imgs = getFaceData()
        M = uvpts.shape[-1]
        K, Rt = face_module3(uvpts, worldpts)

        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        plotHistogram(K[:, 0, 0], save=True, filename='tmp/f1_all.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2_all.png')
        plotHistogram(np.absolute(K[:, 0, 0]), save=True, filename='tmp/f1.png')
        plotHistogram(np.absolute(K[:, 1, 1]), save=True, filename='tmp/f2.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename='tmp')

    elif args.mode == 'face_biwi':
        subject='01'
        uvpts, worldpts,imgs = getFaceDataBIWI(subject,maxangle=30,lmrange=(0,68))

        calib_file = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}/rgb.cal"
        K_gt = util.readBIWICalibration(calib_file)

        # check number of views
        M = uvpts.shape[-1]
        if M < 3:
            print(f"number of valid views is: {M}")
            print("need at least 3")
            quit()
        else:
            K, Rt, _ = face_module3(uvpts, worldpts)

        error = np.absolute(K_gt - np.mean(K,axis=0))
        print(error)

        if not os.path.exists('tmp'):
            os.mkdir('tmp')

        plotHistogram(K[:, 0, 0], save=True, filename='tmp/f1.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename='tmp')

    elif args.mode == 'getpose':
        subject_ids = [f"{i+1:02d}" for i in range(24)]
        total_errors = []

        for n, subject in enumerate(subject_ids):
            calib_file = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}/rgb.cal"
            K_gt = util.readBIWICalibration(calib_file)

            _,_,file_names = getFaceDataBIWI(subject, maxangle=30)
            file_names.sort()
            outdir = f"tmp/biwi/{subject}"
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            for f in file_names:
                copy(f,outdir)
            with open('out.txt', 'a') as fout:
                for f in file_names:
                    fout.write(f + '\n')

    elif args.mode == 'masa_only3':
        uvpts, worldpts, imgs, gt_depth, gt_files = getFaceDataMasa2()

        M = uvpts.shape[-1]
        K, Rt, errors = face_module3(uvpts, worldpts)

        proj = np.matmul(Rt, np.expand_dims(worldpts.T,axis=0))
        depth = np.linalg.norm(proj,axis=1)
        mean_depth = np.mean(depth,axis=1)

        depth_difference = []
        indices = np.arange(gt_files.shape[0])
        for i,f in enumerate(imgs):
            id = indices[gt_files == f]
            if len(id) == 0:
                continue
            else:
                diff = np.absolute(mean_depth[i] - gt_depth[id])
                depth_difference.append(diff)

        plotHistogram(np.array(depth_difference), save=True, filename='tmp/depth_diff.png')
        for i,f in enumerate(imgs):
            img = imageio.imread(f)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(depth_difference[i])
            cv2.putText(img, text, (40,40), font, 1, (0,0,0),2)
            plt.imshow(img)
            outfile = f"{i:02d}_" + os.path.basename(f)
            plt.savefig(os.path.join('tmp',outfile))

        print(depth_difference)
        print(np.mean(depth_difference))
        quit()

        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        plotHistogram(K[:, 0, 0], save=True, filename='tmp/f1_all.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2_all.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename='tmp')

    elif args.mode == 'face_biwi_gtlm2d':
        subject = '02'
        uvpts, worldpts,imgs = util.getFaceDataBIWI_manual(subject,maxangle=30)

        calib_file = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}/rgb.cal"
        K_gt = util.readBIWICalibration(calib_file)

        # check number of views
        M = uvpts.shape[-1]
        if M < 3:
            print(f"number of valid views is: {M}")
            print("need at least 3")
            quit()
        else:
            K, Rt, _ = face_module3(uvpts, worldpts)

        error = np.absolute(K_gt - np.mean(K,axis=0))
        print(error)
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        plotHistogram(K[:, 0, 0], save=True, filename='tmp/f1.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename='tmp')

    elif args.mode == 'face_biwi_full':
        subject='01'
        for i in range(15):
            angle = 5 + 100/20 * i

            # save all results for angle + lm range to folder
            lmrange = (0,68)
            uvpts,worldpts,imgs = getFaceDataBIWI(subject, maxangle=angle,lmrange=lmrange)
            K, Rt = face_module3(uvpts, worldpts)
            outfile = 'tmp/biwi_full_' + ("%03d" % angle) + '_lmrange_' + ("%02d-%02d" % lmrange)
            if not os.path.exists(outfile):
                os.mkdir(outfile)
            plotHistogram(np.absolute(K[:, 0, 0]), save=True, filename=os.path.join(outfile,'f1.png'))
            plotHistogram(np.absolute(K[:, 1, 1]), save=True, filename=os.path.join(outfile,'f2.png'))
            plotHistogram(K[:, 0, 2], save=True, filename=os.path.join(outfile,'uc.png'))
            plotHistogram(K[:, 1, 2], save=True, filename=os.path.join(outfile,'vc.png'))
            plotHistogram(K[:, 0, 1], save=True, filename=os.path.join(outfile,'skew.png'))
            save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename=outfile)
            plt.close(fig='all')

            # lm range 2
            lmrange = (27,68)
            uvpts,worldpts,imgs = getFaceDataBIWI(subject, maxangle=angle,lmrange=lmrange)
            K, Rt = face_module3(uvpts, worldpts)
            outfile = 'tmp/biwi_full_' + ("%03d" % angle) + '_lmrange_' + ("%02d-%02d" % lmrange)
            if not os.path.exists(outfile):
                os.mkdir(outfile)
            plotHistogram(np.absolute(K[:, 0, 0]), save=True, filename=os.path.join(outfile,'f1.png'))
            plotHistogram(np.absolute(K[:, 1, 1]), save=True, filename=os.path.join(outfile,'f2.png'))
            plotHistogram(K[:, 0, 2], save=True, filename=os.path.join(outfile,'uc.png'))
            plotHistogram(K[:, 1, 2], save=True, filename=os.path.join(outfile,'vc.png'))
            plotHistogram(K[:, 0, 1], save=True, filename=os.path.join(outfile,'skew.png'))
            save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename=outfile)
            plt.close(fig='all')

            # lm range 3
            lmrange = (27,48)
            uvpts,worldpts,imgs = getFaceDataBIWI(subject, maxangle=angle,lmrange=lmrange)
            K, Rt = face_module3(uvpts, worldpts)
            outfile = 'tmp/biwi_full_' + ("%03d" % angle) + '_lmrange_' + ("%02d-%02d" % lmrange)
            if not os.path.exists(outfile):
                os.mkdir(outfile)
            plotHistogram(np.absolute(K[:, 0, 0]), save=True, filename=os.path.join(outfile,'f1.png'))
            plotHistogram(np.absolute(K[:, 1, 1]), save=True, filename=os.path.join(outfile,'f2.png'))
            plotHistogram(K[:, 0, 2], save=True, filename=os.path.join(outfile,'uc.png'))
            plotHistogram(K[:, 1, 2], save=True, filename=os.path.join(outfile,'vc.png'))
            plotHistogram(K[:, 0, 1], save=True, filename=os.path.join(outfile,'skew.png'))
            save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename=outfile)
            plt.close(fig='all')

    elif args.mode == 'checkerboard':
        checkerboard_module()
    elif args.mode == 'checkerboard2':
        checkerboard_module2()
    elif args.mode == 'checkerboard3':
        checkerboard_module3()

    elif args.mode == 'kitti_synth':
        M = 100
        kitti_loader = dataloader.DataLoader()
        uvcoord, worldcoord = kitti_loader.readN(M)
        uvcoord = np.concatenate((uvcoord,np.ones((M,8,1))),axis=2)
        worldcoord = np.concatenate((worldcoord,np.ones((M,8,1))),axis=2)
        K, Rt = calibrate(uvcoord,worldcoord)
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        plotHistogram(K[:, 0, 0], filename='tmp/f1.png')
        plotHistogram(K[:, 1, 1], save=True, filename='tmp/f2.png')
        plotHistogram(K[:, 0, 2], save=True, filename='tmp/uc.png')
        plotHistogram(K[:, 1, 2], save=True, filename='tmp/vc.png')
        plotHistogram(K[:, 0, 1], save=True, filename='tmp/skew.png')
        #save2DReprojection(K, Rt, uvpts, worldpts,imgs, filename='tmp')

    else:
        print('please use one of the following modes')
        print('face')
        print('face2')
        print('face3')
        print('face4')
        print('checkerboard')
        print('checkerboard2')
