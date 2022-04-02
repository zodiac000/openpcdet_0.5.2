'''
Author: your name
Date: 2022-01-28 14:16:17
LastEditTime: 2022-01-29 06:13:48
LastEditors: Please set LastEditors
Description: 绘制紧凑框和联想框的逐帧结果
FilePath: /OpenPCDet/tools/scripts/process_boundary.py
'''
import os
import pickle
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
import numpy as np
from pcdet.utils.object3d_multi_frames import Object3d
from tqdm import tqdm
def process_tight_box(points_norm, box3d_lidar, fig=None):
    
    box3d_lidar_norm = box3d_lidar.copy()
    box3d_lidar_norm[:3] = 0

    rotation = -box3d_lidar_norm[-1]
    points_norm_rot = common_utils.rotate_points_along_z(points_norm[np.newaxis, :, :], np.array([rotation]))[0]
    
    line_str   = " ".join(str(i) for i in box3d_lidar_norm)

    obj3d_src  = Object3d(line_str)
    obj3d_corners_src = obj3d_src.generate_corners3d()

    if len(points_norm) == 0:
        return dict(box3d_lidar=box3d_lidar, \
                obj3d_corners_src=obj3d_corners_src,\
                box3d_lidar_des=box3d_lidar.copy(), \
                obj3d_corners_des=obj3d_corners_src, \
                obj3d_centroid = box3d_lidar[:2],
                is_valid=False,
                )
    centroid = np.mean(points_norm[:,:2], axis=0)  # 质心
    minx, maxx = np.min(points_norm_rot[:,0]), np.max(points_norm_rot[:,0])
    miny, maxy = np.min(points_norm_rot[:,1]), np.max(points_norm_rot[:,1])

    length  = maxx - minx
    width = maxy - miny
    
    center_x = 0.5*(minx + maxx) # 紧缩后的中心
    center_y = 0.5*(miny + maxy)

    center_rot = np.array([center_x, center_y, 0, 0]).reshape((1,4))
    tight_center_2_base = common_utils.rotate_points_along_z(center_rot[np.newaxis, :, :], np.array([-rotation]))[0][0]
    
    box3d_lidar_des = [tight_center_2_base[0], tight_center_2_base[1], box3d_lidar_norm[2], length, width, box3d_lidar_norm[-2], box3d_lidar_norm[-1]]
    line_str   = " ".join(str(i) for i in box3d_lidar_des)
    obj3d_des  = Object3d(line_str)
    obj3d_corners_des = obj3d_des.generate_corners3d()    

    points = points_norm[:,:3] + box3d_lidar[:3].reshape((1,3))
    obj3d_corners_src[:,:3] += box3d_lidar[:3].reshape((1,3))
    obj3d_corners_des[:,:3] += box3d_lidar[:3].reshape((1,3))
    box3d_lidar_des[:3]   += box3d_lidar[:3]
    centroid += box3d_lidar[:2]
    if fig is not None:
        ax1 = plt.subplot(121)
        ax1.plot(-points_norm[:,1], points_norm[:,0],'b.')
        ax1.plot(-points_norm_rot[:,1], points_norm_rot[:,0],'r.')
        ax1.axis('equal')
        ax2 = plt.subplot(122)
        ax2.plot(-points[:,1], points[:,0],'b.')
        ax2.plot(-obj3d_corners_src[:,1], obj3d_corners_src[:,0],'b-')
        ax2.plot(-obj3d_corners_des[:,1], obj3d_corners_des[:,0],'r-')
        ax2.plot(-box3d_lidar[1], box3d_lidar[0], 'b+')
        ax2.plot(-box3d_lidar_des[1], box3d_lidar_des[0], 'r+')
        ax2.plot(-centroid[1], centroid[0], 'k+')
        ax2.axis('equal')
    
    return dict(box3d_lidar=box3d_lidar, \
                obj3d_corners_src=obj3d_corners_src,\
                box3d_lidar_des=box3d_lidar_des, \
                obj3d_corners_des=obj3d_corners_des, \
                obj3d_centroid = centroid,
                is_valid=True,
                )

def plot_trajactory(tight_box_infos, save_filename=None):
    
    obj3d_centroids  = []
    box3d_lidar_dess = []
    box3d_lidar_srcs = []
    for tight_box_info in tight_box_infos:
        obj3d_centroids.append(tight_box_info["obj3d_centroid"])
        box3d_lidar_dess.append(tight_box_info["box3d_lidar_des"])
        box3d_lidar_srcs.append(tight_box_info["box3d_lidar"])
    
    box3d_lidar_srcs = np.array(box3d_lidar_srcs)    
    box3d_lidar_dess = np.array(box3d_lidar_dess)
    obj3d_centroids  = np.array(obj3d_centroids)

    center_src_speed   = (box3d_lidar_srcs[1:,:2] - box3d_lidar_srcs[:-1,:2])/0.1
    center_tight_speed = (box3d_lidar_dess[1:,:2] - box3d_lidar_dess[:-1,:2])/0.1
    centriod_speed     = (obj3d_centroids[1:,:2] - obj3d_centroids[:-1,:2])/0.1

    if save_filename is not None:
        fig = plt.figure()
        ax = plt.subplot(121)
        seq_idxs = np.arange(len(box3d_lidar_srcs))
        ax.plot(seq_idxs, box3d_lidar_srcs[:,0],'r')
        ax.plot(seq_idxs, box3d_lidar_dess[:,0],'g')
        ax.plot(seq_idxs, obj3d_centroids[:,0],'b')
        ax.legend(["box3d_center_src", "box3d_center_tight", "box3d_centriod"])
        ax.set_title("x-depth")
        ax = plt.subplot(122)
        seq_idxs = np.arange(1,len(box3d_lidar_srcs))
        ax.plot(seq_idxs, center_src_speed[:,0],'r')
        ax.plot(seq_idxs, center_tight_speed[:,0],'g')
        ax.plot(seq_idxs, centriod_speed[:,0],'b')
        ax.legend(["center_src_speed", "center_tight_speed", "centriod_speed"])
        ax.set_title("x-speed")
        plt.savefig(save_filename)
if __name__ == '__main__':
    pass

    data_root = "/data7/zlh/lidar_detection/OpenPCDet/tools/debug_track"
    gt_pkl_file = "%s/track_info.pkl"%data_root
    gt_pkl_tight_box_file = "%s/track_tight_box_info.pkl"%data_root

    save_dir = "tools/work_dirs/debug_track/point_tight"
    os.makedirs(save_dir, exist_ok=True)

    with open(gt_pkl_file, "rb") as fid:
        gt_infos = pickle.load(fid)
    
    for uuid, track_datas in tqdm(gt_infos["Car"].items()):
        if uuid != '418b7f15-e6c4-40fe-b440-ff747b22430b':
            continue
        uuid_dir = save_dir+"/"+uuid
        os.makedirs(save_dir+"/"+uuid+"/seqs", exist_ok = True)

        tight_box_infos = []
        for i, (box3d_lidar, gt_points) in enumerate(tqdm(zip(track_datas["box3d"],track_datas["gt_points"]))):
            # if i > 0:
            #     break
            fig = plt.figure()
            tight_box_info = process_tight_box(gt_points, np.array(box3d_lidar), fig)            
            tight_box_infos.append(tight_box_info)
            
            save_filename = "%s/seqs/%04d.png"%(uuid_dir, i)
            plt.savefig(save_filename)

        plot_trajactory(tight_box_infos, save_filename="%s/trajactory.png"%uuid_dir)

        track_datas.update(tight_box_infos=tight_box_infos)
        break

    # with open(gt_pkl_tight_box_file, "wb") as fid:
    #     pickle.dump(gt_infos, fid)