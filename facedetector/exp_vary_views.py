
# native imports
import random
import os

# opensource imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
import util
import main

#######################################################################################
# vary the number of views when doing camera calibration on the biwi kinect dataset
# check calibration performance across all subjects and see if it makes sense
# make sure face module 3 doesn't open any figures with pyplot otherwise this will crash
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
if __name__ == '__main__':

    subject_ids = [f"{i+1:02d}"  for i in range(24)]
    total_errors = []

    for n,subject in enumerate(subject_ids):
        calib_file = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{subject}/rgb.cal"
        K_gt = util.readBIWICalibration(calib_file)

        # angles, lmranges, samplesize, predictions
        pred = np.zeros((10,3,10,6))
        for l in range(10):
            angle = 20 + 100/20 * l

            lmrange1 = (0,68)
            lmrange2 = (27,68)
            lmrange3 = (27,48)
            uvpts,worldpts,imgs = main.getFaceDataBIWI(subject, maxangle=angle)
            M = len(imgs)
            for i, lmrange in enumerate([lmrange1,lmrange2,lmrange3]):
                for j in range(10):
                    samplesize = int(M * (j+1)/10)
                    indices = random.sample(range(M),samplesize)
                    K, Rt, errors = main.face_module3(uvpts[lmrange[0]:lmrange[1],:,indices], worldpts[lmrange[0]:lmrange[1]])
                    pred[l,i,j] = (np.mean(K[:,0,0]),np.mean(K[:,0,1]),np.mean(K[:,0,2]),np.mean(K[:,1,1]), np.mean(K[:,1,2]),np.mean(errors))

        pred[:,:,:,0] -= K_gt[0,0]
        pred[:,:,:,1] -= K_gt[0,1]
        pred[:,:,:,2] -= K_gt[0,2]
        pred[:,:,:,3] -= K_gt[1,1]
        pred[:,:,:,4] -= K_gt[1,2]
        pred[:,:,:,5] -= 0
        error = np.absolute(pred)
        total_errors.append(error.copy())

    total_errors = np.stack(total_errors,axis=0)

    # from all subjects
    average_errors = np.mean(total_errors,axis=0)
    #errors = np.concatenate((total_errors,np.expand_dims(average_errors,axis=0)), axis=0)

    #mean_error = np.mean(total_errors,axis=0)
    #np.expand_dims(np.std(errors,axis=2),axis=2)
    #std_error = np.expand_dims(np.std(errors,axis=2),axis=2)
    #errors = np.concatenate((errors,std_error),axis=2)

    samples = [str((i+1)/10) for i in range(10)]
    #subject = [f"subject {i+1:02d}" for i in range(24)]
    ylabel = [str(20 + 100/20 * l) for l in range(10)]

    out_dir = ["lm_0-68","lm_27-68","lm_48-68"]
    for i in range(3):
        if not os.path.exists(out_dir[i]):
            os.mkdir(out_dir[i])
        #util.createHeatMap(errors[:,i,:,0], xlabels=samples, ylabels=subject, title="Error F1",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"f1_error.png"))
        #util.createHeatMap(errors[:,i,:,1], xlabels=samples, ylabels=subject, title="Error Skew",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"skew_error.png"))
        #util.createHeatMap(errors[:,i,:,2], xlabels=samples, ylabels=subject, title="Error Uc",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"uc_error.png"))
        #util.createHeatMap(errors[:,i,:,3], xlabels=samples, ylabels=subject, title="Error F2",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"f2_error.png"))
        #util.createHeatMap(errors[:,i,:,4], xlabels=samples, ylabels=subject, title="Error Vc",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"vc_error.png"))
        #util.createHeatMap(errors[:,i,:,5], xlabels=samples, ylabels=subject, title="Reprojection Error",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"reproj_error.png"))

        util.createHeatMap(average_errors[:,i,:,0], xlabels=samples, ylabels=ylabel, title="Error F1",xtitle="Samples", ytitle="max pose", save=True, fileout=os.path.join(out_dir[i],"f1_error.png"))
        util.createHeatMap(average_errors[:,i,:,1], xlabels=samples, ylabels=ylabel, title="Error Skew",xtitle="Samples", ytitle="max pose", save=True, fileout=os.path.join(out_dir[i],"skew_error.png"))
        util.createHeatMap(average_errors[:,i,:,2], xlabels=samples, ylabels=ylabel, title="Error Uc",xtitle="Samples", ytitle="max pose", save=True, fileout=os.path.join(out_dir[i],"uc_error.png"))
        util.createHeatMap(average_errors[:,i,:,3], xlabels=samples, ylabels=ylabel, title="Error F2",xtitle="Samples", ytitle="max pose", save=True, fileout=os.path.join(out_dir[i],"f2_error.png"))
        util.createHeatMap(average_errors[:,i,:,4], xlabels=samples, ylabels=ylabel, title="Error Vc",xtitle="Samples", ytitle="max pose", save=True, fileout=os.path.join(out_dir[i],"vc_error.png"))
        util.createHeatMap(average_errors[:,i,:,5], xlabels=samples, ylabels=ylabel, title="Reprojection Error",xtitle="Samples", ytitle="max pose", save=True, fileout=os.path.join(out_dir[i],"reproj_error.png"))
        #util.createHeatMap(std_error[:,i,0], xlabels=samples, ylabels=subject, title="Error STD F1",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"f1_std.png"))
        #util.createHeatMap(std_error[:,i,1], xlabels=samples, ylabels=subject, title="Error STD Skew",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"skew_std.png"))
        #util.createHeatMap(std_error[:,i,2], xlabels=samples, ylabels=subject, title="Error STD Uc",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"uc_std.png"))
        #util.createHeatMap(std_error[:,i,3], xlabels=samples, ylabels=subject, title="Error STD F2",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"f2_std.png"))
        #util.createHeatMap(std_error[:,i,4], xlabels=samples, ylabels=subject, title="Error STD Vc",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"vc_std.png"))
        #util.createHeatMap(std_error[:,i,5], xlabels=samples, ylabels=subject, title="Reprojection Error STD",xtitle="Samples", ytitle="subject name", save=True, fileout=os.path.join(out_dir[i],"reproj_std.png"))

