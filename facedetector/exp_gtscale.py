

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

    subject_ids = [f"{i+1:02d}"  for i in range(24)]
    total_errors = []

    for n,subject in enumerate(subject_ids):
        calib_file = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}/rgb.cal"
        K_gt = util.readBIWICalibration(calib_file)

        K1 = []
        R1 = []
        K2 = []
        R2 = []
        K3 = []
        R3 = []
        for i in range(10):
            angle = 10 + 100/20 * i

            lmrange = (0,68)
            uvpts,worldpts,imgs = util.getFaceDataBIWI_gtsize(subject, maxangle=angle,lmrange=lmrange)
            K, Rt, errors = main.face_module4(uvpts, worldpts)
            K1.append((np.mean(K[:,0,0]),np.mean(K[:,0,1]),np.mean(K[:,0,2]),np.mean(K[:,1,1]), np.mean(K[:,1,2]),np.mean(errors)))
            R1.append(Rt)

            lmrange = (27,68)
            uvpts,worldpts,imgs = util.getFaceDataBIWI_gtsize(subject, maxangle=angle,lmrange=lmrange)
            K, Rt, errors = main.face_module4(uvpts, worldpts)
            K2.append((np.mean(K[:,0,0]),np.mean(K[:,0,1]),np.mean(K[:,0,2]),np.mean(K[:,1,1]), np.mean(K[:,1,2]), np.mean(errors)))
            R2.append(Rt)

            lmrange = (27,48)
            uvpts,worldpts,imgs = util.getFaceDataBIWI_gtsize(subject, maxangle=angle,lmrange=lmrange)
            K, Rt, errors = main.face_module4(uvpts, worldpts)
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
        total_errors.append(K_all)
        total_errors.append(K_all.copy())

    total_errors = np.stack(total_errors,axis=0)
    mean_error = np.mean(total_errors,axis=0)
    std_error = np.std(total_errors,axis=0)

    angles = [str(10 + 5*i) for i in range(10)]
    lm_labels = ["lm 0-68", "lm 27-68", "lm 27-48"]

    util.createHeatMap(mean_error[:,:,0], xlabels=angles, ylabels=lm_labels, title="Error F1",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="f1_error.png")
    util.createHeatMap(mean_error[:,:,1], xlabels=angles, ylabels=lm_labels, title="Error Skew",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="skew_error.png")
    util.createHeatMap(mean_error[:,:,2], xlabels=angles, ylabels=lm_labels, title="Error Uc",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="uc_error.png")
    util.createHeatMap(mean_error[:,:,3], xlabels=angles, ylabels=lm_labels, title="Error F2",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="f2_error.png")
    util.createHeatMap(mean_error[:,:,4], xlabels=angles, ylabels=lm_labels, title="Error Vc",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="vc_error.png")
    util.createHeatMap(mean_error[:,:,5], xlabels=angles, ylabels=lm_labels, title="Reprojection Error",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="reproj_error.png")

    util.createHeatMap(std_error[:,:,0], xlabels=angles, ylabels=lm_labels, title="Error STD F1",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="f1_std.png")
    util.createHeatMap(std_error[:,:,1], xlabels=angles, ylabels=lm_labels, title="Error STD Skew",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="skew_std.png")
    util.createHeatMap(std_error[:,:,2], xlabels=angles, ylabels=lm_labels, title="Error STD Uc",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="uc_std.png")
    util.createHeatMap(std_error[:,:,3], xlabels=angles, ylabels=lm_labels, title="Error STD F2",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="f2_std.png")
    util.createHeatMap(std_error[:,:,4], xlabels=angles, ylabels=lm_labels, title="Error STD Vc",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="vc_std.png")
    util.createHeatMap(std_error[:,:,5], xlabels=angles, ylabels=lm_labels, title="Reprojection Error STD",xtitle="Max Pose (degrees)", ytitle="Landmark Subset", save=True, fileout="reproj_std.png")
