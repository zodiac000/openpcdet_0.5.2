'''
Author: your name
Date: 2022-01-28 11:43:22
LastEditTime: 2022-01-29 06:05:46
LastEditors: Please set LastEditors
Description: 生成track_info.pkl 文件
FilePath: /OpenPCDet/tools/scripts/show_trategy.py
'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def gen_tracklist(label_lists):
    track_dict = {'Car':{}, 'Pedestrian':{}, 'Cyclist':{}}
    for label in label_lists:
        frame_id = os.path.basename(label).split('.')[0]
        with open(label, "r") as fid:
            lines = fid.readlines()
        for line in lines:            
            arr = line.strip().split()
            uuid = arr[-1]
            type_name = arr[-2]
            if type_name not in track_dict:
                continue
            arr_float = [float(x) for x in arr[:-3]]
            if uuid not in track_dict[type_name]:               
                
                track_dict[type_name][uuid] = {"box3d":[],"frame_id":[]}
                track_dict[type_name][uuid]["box3d"] = [arr_float]
                track_dict[type_name][uuid]["frame_id"] = [frame_id]
            else:    
                track_dict[type_name][uuid]["box3d"].append(arr_float)
                track_dict[type_name][uuid]["frame_id"].append(frame_id)
    return track_dict

def gen_track_label_file(track_root, seq_id):
    pass
    seq_id_fileformat = track_root + "/" + seq_id + "_"
    files = glob.glob(seq_id_fileformat + "*.txt")
    sort_files = sorted(files, key=lambda x:x.split(seq_id_fileformat)[1])
    
    return sort_files
# def test_kedu():
#     x_major_locator=MultipleLocator(1)
#     y_major_locator=MultipleLocator(10)
#     ax=plt.gca()
#     #ax为两条坐标轴的实例
#     ax.xaxis.set_major_locator(x_major_locator)
#     #把x轴的主刻度设置为1的倍数
#     ax.yaxis.set_major_locator(y_major_locator)

def process_roi_points(points, gt_boxes):
    import torch
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
    
    num_obj = len(gt_boxes)
    gt_points_list = []
    for i in range(num_obj):
        # filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
        # filepath = database_save_path / filename
        gt_points  = points[point_indices[i] > 0]
        gt_points[:, :3] -= gt_boxes[i, :3]
        # with open(filepath, 'w') as f:
        #     gt_points.tofile(f)
        gt_points_list.append(gt_points)
    return gt_points_list

def process_track_infos(track_datas_cl_t, velody_root):
    
    
    for type_name in tqdm(track_datas_cl_t):
        track_cls_datas = track_datas_cl_t[type_name]
        
        for key in tqdm(track_cls_datas):
            track_datas = track_cls_datas[key]
            # if key != "d7129434-b9e5-4afa-8844-861ac21042d0":
            #     continue
            track_gt_points = []
            for box3d, frame_id in zip(track_datas["box3d"], track_datas["frame_id"]):
                lidar_file  = os.path.join(velody_root, frame_id+".bin")        
                lidar_data  = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

                gt_boxes       = np.array([box3d])
                gt_points_list = process_roi_points(lidar_data, gt_boxes)
                track_gt_points.append(gt_points_list[0])
            track_datas.update(gt_points=track_gt_points)
    return track_datas

if __name__ == '__main__':
    pass

    track_root = "/data8/duzhe/dataset/untouch/lidar_datasets/pandaset/training/label_track"
    velody_root = "/data8/duzhe/dataset/untouch/lidar_datasets/pandaset/training/velodyne"

    save_dir   = "./debug_track"

    seq_id = "002"

    os.makedirs(save_dir, exist_ok = True)
    frame_time = 0.1
    track_label_files = gen_track_label_file(track_root, seq_id)
    track_dict = gen_tracklist(track_label_files)
    
    process_track_infos(track_dict, velody_root)

        # label_traj = np.array(box3d)

        # fig_file = os.path.join(save_dir,key+"xy.png")
        
        # plt.figure()
        # plt.plot(np.arange(len(label_traj)), label_traj[:, 0],'r')
        # # plt.plot(label_traj[:, 0], label_traj[:, 1],'r')
        # plt.savefig(fig_file)

    with open(os.path.join(save_dir, "track_info.pkl"), 'wb') as fid:
        pickle.dump(track_dict, fid)

    print("end")