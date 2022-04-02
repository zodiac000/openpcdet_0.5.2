'''
Author: your name
Date: 2022-01-28 14:16:17
LastEditTime: 2022-01-29 04:50:22
LastEditors: Please set LastEditors
Description: 生成紧凑框的pkl文件
FilePath: /OpenPCDet/tools/scripts/process_boundary.py
'''
import os
import pickle
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
import numpy as np
from pcdet.utils.object3d_untouch import Object3d
from tqdm import tqdm

def process_tight_box(points_norm, box3d_lidar, fig=None):
    pass    

    box3d_lidar_norm = box3d_lidar.copy()
    box3d_lidar_norm[:3] = 0

    centroid = np.mean(points_norm[:,:2], axis=0)  # 质心

    rotation = -box3d_lidar_norm[-1]
    points_norm_rot = common_utils.rotate_points_along_z(points_norm[np.newaxis, :, :], np.array([rotation]))[0]
    
    line_str   = " ".join(str(i) for i in box3d_lidar_norm)

    obj3d_src  = Object3d(line_str)
    obj3d_corners_src = obj3d_src.generate_corners3d()

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
                )
if __name__ == '__main__':
    pass

    data_root = "./debug_track/"
    gt_pkl_file = "%s/track_info.pkl"%data_root
    gt_pkl_tight_box_file = "%s/track_tight_box_info.pkl"%data_root

    save_dir = "debug_track/point_tight"
    os.makedirs(save_dir, exist_ok=True)

    data_root = "/data7/zlh/lidar_detection/OpenPCDet/data/"
    gt_pkl_file = "%s/pandaset/untouch_dbinfos_train.pkl"%data_root

    
    with open(gt_pkl_file, "rb") as fid:
        gt_infos = pickle.load(fid)
    seq = "/002_"
    tight_box_infos = []
    for i,gt_info in enumerate(tqdm(gt_infos["Car"])):
        
        box3d_lidar = gt_info["box3d_lidar"]
        path = gt_info["path"]
        if seq not in path:
            continue
        lidar_file = os.path.join(data_root, path)
        gt_points  = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        fig = plt.figure()
        tight_box_info = process_tight_box(gt_points, np.array(box3d_lidar), fig)
        
        tight_box_infos.append(tight_box_info)

        save_filename = "%s/%04d.png"%(save_dir, i)
        plt.savefig(save_filename)
            
    track_datas.update(tight_box_infos=tight_box_infos)
    with open(gt_pkl_tight_box_file, "wb") as fid:
        pickle.dump(gt_infos, fid)    