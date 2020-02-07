
import os

import imageio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2

import util

###########################################################################################################

subjects = [f"{i+1:02d}" for i in range(24)]
for sub in subjects:

    # get uvcoord predictions and world pt predictions
    uvcoord, worldpts, valid_paths = util.getFaceDataBIWI(sub, maxangle=90)

    # get kinect data
    root_dir = f'/home/huynshen/data/kinect_head_pose_db/hpdb/{sub}'
    M = len(valid_paths)
    N = 68
    worldcoord = np.ones((M,N,4))

    #READ THE RGB CALIBRATION FILE
    rgb_calibfile = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{sub}/rgb.cal"
    rgb_I, rgb_R, rgb_T2 = util.readBIWICalib(rgb_calibfile)

    # for each valid path with face
    for i,f in enumerate(valid_paths):
        # create output folder
        outdir = os.path.join('tmp',sub)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        rgbfile = f
        xyzfile = f.replace('rgb','xyz').replace('png','mat').replace('hpdb_copy','hpdb')

        # load the data kinect data with camera info
        data = scipy.io.loadmat(xyzfile)
        p3d = data['pts3d']
        p2d = data['pts2d']
        dmap = data['dmap']
        mask = dmap[:,:,-1] != -1

        # find closest landmarks in 3d shape that projects to the 2d location
        landmarks = uvcoord[:,:,i].astype(np.uint32)
        lmidx = []
        for pt in landmarks:
            x,y = util.findValidPoint(mask, (pt[0],pt[1]))
            idx = dmap[y,x][-1]
            lmidx.append(int(idx))

        # perform rigid transformation from worldpts to 3d point cloud data
        ptcloud = p3d[:,lmidx].T
        error, transpts, _ = util.procrustes(ptcloud,worldpts[:,:3])

        # project aligned 3d points of our 3d model back to the image under known intrinsics
        reproj = rgb_I.dot(transpts.T).T
        reproj = reproj / np.expand_dims(reproj[:,-1],axis=1)
        error_2d = np.mean(np.linalg.norm(reproj - landmarks,axis=1))

        # visualize corrected 2d landmarks
        outfile1 = os.path.join(outdir, os.path.basename(f))
        outfile2 = os.path.join(outdir, os.path.basename(f).replace('rgb', 'gtlm2d').replace('png','mat'))
        canvas = plt.imread(f) * 255
        canvas = util.drawpts(canvas, reproj)
        canvas = util.drawpts(canvas, landmarks,color=[0,255,0])
        cv2.imshow('img', cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        imageio.imwrite(outfile1, canvas.astype(np.uint8))
        gtlm2d = {'lm2d': reproj}
        scipy.io.savemat(outfile2,gtlm2d)



