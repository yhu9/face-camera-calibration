

import os

import scipy.io

import util

if __name__ == '__main__':


    subjects = [f"{i+1:02d}" for i in range(24)]
    for sub in subjects:
        rgb_calibfile = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{sub}/rgb.cal"
        depth_calibfile = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{sub}/depth.cal"
        root_dir = f"/home/huynshen/data/kinect_head_pose_db/hpdb/{sub}"

        files = os.listdir(root_dir)
        rgb_files = [os.path.join(root_dir,f) for f in files if f[-4:] == '.png']
        for f in rgb_files:
            depth_file = f.replace('rgb.png','depth.bin')
            dmap, pts3d, pts2d = util.readDepthData(depth_file, rgb_calibfile, depth_calibfile)
            outfile = f.replace('rgb.png','xyz.mat')
            data = {'dmap': dmap, 'pts3d': pts3d, 'pts2d': pts2d}
            scipy.io.savemat(outfile, data)

            print(f"process: {depth_file} ----> {outfile}")

