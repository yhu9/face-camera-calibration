
# native imports
import os

# opensource imports
import cv2
import imageio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# local imports
import util
import main

##################################################################################################

if __name__ == '__main__':

    # manual 2d landmark only available for subject 02 for now
    #print(uvpts.shape)

    subject_ids = [f"{i+1:02d}"  for i in range(24)]
    total_errors = []


    # for each subject
    for n,subject in enumerate(subject_ids):
        # uuhhhh I was too lazy to fix figuring out how to determine how the data cooresponds to the file
        _,_,all_paths = util.getFaceDataBIWI(subject, maxangle=90, lmrange=(0,68))

        # load gt calibration data per subject
        calib_file = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}/rgb.cal"
        K_gt = util.readBIWICalibration(calib_file)
        K1 = []
        R1 = []
        K2 = []
        R2 = []
        K3 = []
        R3 = []

        # sample 20 different magnitudes of noise from uniform/gaussian distribution
        # max pose angle is 30 and all landmarks are used
        angle = 30
        lmrange = (0,68)
        for i in range(21):
            # predicted 2d landmark from different network of predicted 3d landmarks
            uvpts,worldpts,valid_paths = util.getFaceDataBIWI(subject, maxangle=angle, lmrange=lmrange)
            noise = np.random.normal(0,i/10,uvpts.shape)
            uvpts = np.clip(uvpts + noise,0,680)
            uvpts[:,-1] = 1
            K, Rt, errors = main.face_module3(uvpts, worldpts)
            K1.append((np.mean(K[:,0,0]),np.mean(K[:,0,1]),np.mean(K[:,0,2]),np.mean(K[:,1,1]), np.mean(K[:,1,2]),np.mean(errors)))
            R1.append(Rt)

            # predicted 2d landmarks from weak perspective 3d model
            # oops I have no idea if the uvpts are ordered according to the correct pose
            uvpts, worldpts = util.getFaceDataBIWI_weak(subject, maxangle=angle, lmrange=lmrange)
            uvpts = np.clip(uvpts + noise,0,680)
            uvpts[:,-1] = 1
            K, Rt, errors = main.face_module3(uvpts, worldpts)
            K2.append((np.mean(K[:,0,0]),np.mean(K[:,0,1]),np.mean(K[:,0,2]),np.mean(K[:,1,1]), np.mean(K[:,1,2]),np.mean(errors)))
            R2.append(Rt)

            # manually labeled 2d landmarks
            # uvpts,worldpts,_ = util.getFaceDataBIWI_manual(subject, maxangle=90)

            # ground truth 2d landmarks with known calibration. perfect 3d 2d correspondence and 2d 2d correspondence
            # get ground truth for files processed by every other method
            uvpts, worldpts = util.getFaceDataBIWI_gt(subject, valid_paths, maxangle=angle, lmrange=lmrange)
            uvpts = np.clip(uvpts + noise, 0, 680)
            uvpts[:,-1] = 1
            K, Rt, errors = main.face_module3(uvpts, worldpts)
            K3.append((np.mean(K[:,0,0]),np.mean(K[:,0,1]),np.mean(K[:,0,2]),np.mean(K[:,1,1]), np.mean(K[:,1,2]), np.mean(errors)))
            R3.append(Rt)

        K1 = np.stack(K1,axis=0)
        K2 = np.stack(K2,axis=0)
        K3 = np.stack(K3,axis=0)
        K_all = np.stack((K1,K2,K3),axis=0)
        K_all[:,:,0] -= K_gt[0,0]
        K_all[:,:,1] -= K_gt[0,1]
        K_all[:,:,2] -= K_gt[0,2]
        K_all[:,:,3] -= K_gt[1,1]
        K_all[:,:,4] -= K_gt[1,2]
        K_all[:,:,5] -= 0
        K_all = np.absolute(K_all)
        total_errors.append(K_all.copy())

    total_errors = np.stack(total_errors,axis=0)
    mean_error = np.mean(total_errors,axis=0)
    std_error = np.std(total_errors,axis=0)

    xlabel = [str(i/10) for i in range(21)]
    ylabel = ["hourglass 2d lm", "weak perspective 2d lm", "GT 2d landmark"]

    util.createHeatMap(mean_error[:,:,0], xlabels=xlabel, ylabels=ylabel, title="Error F1",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="f1_error.png")
    util.createHeatMap(mean_error[:,:,1], xlabels=xlabel, ylabels=ylabel, title="Error Skew",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="skew_error.png")
    util.createHeatMap(mean_error[:,:,2], xlabels=xlabel, ylabels=ylabel, title="Error Uc",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="uc_error.png")
    util.createHeatMap(mean_error[:,:,3], xlabels=xlabel, ylabels=ylabel, title="Error F2",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="f2_error.png")
    util.createHeatMap(mean_error[:,:,4], xlabels=xlabel, ylabels=ylabel, title="Error Vc",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="vc_error.png")
    util.createHeatMap(mean_error[:,:,5], xlabels=xlabel, ylabels=ylabel, title="Reprojection Error",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="reproj_error.png")

    util.createHeatMap(std_error[:,:,0], xlabels=xlabel, ylabels=ylabel, title="Error STD F1",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="f1_std.png")
    util.createHeatMap(std_error[:,:,1], xlabels=xlabel, ylabels=ylabel, title="Error STD Skew",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="skew_std.png")
    util.createHeatMap(std_error[:,:,2], xlabels=xlabel, ylabels=ylabel, title="Error STD Uc",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="uc_std.png")
    util.createHeatMap(std_error[:,:,3], xlabels=xlabel, ylabels=ylabel, title="Error STD F2",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="f2_std.png")
    util.createHeatMap(std_error[:,:,4], xlabels=xlabel, ylabels=ylabel, title="Error STD Vc",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="vc_std.png")
    util.createHeatMap(std_error[:,:,5], xlabels=xlabel, ylabels=ylabel, title="Reprojection Error STD",xtitle="Noise Magnitude (pixels)", ytitle="Landmark Method", save=True, fileout="reproj_std.png")


