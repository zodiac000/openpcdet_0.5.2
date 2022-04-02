'''
Author: lanka
Date: 2022-01-24 21:33:59
LastEditTime: 2022-01-25 04:00:07
LastEditors: lanka
Description: 
FilePath: /OpenPCDet/data7/zlh/lidar_detection/openpcdet-v0.5.2/pcdet/datasets/kitti/kitti_object_eval_python/eval_v3.py
copyright: untouch
'''
import io as sysio
from tkinter import N
from tkinter.messagebox import NO

import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


import numpy as np

from rotate_iou import rotate_iou_gpu_eval

from kitti_common import get_label_annos_str
import matplotlib.pyplot as plt
import pickle
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
import os
from pathlib import Path

@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0) # 最终并非一定能达到1.0
        
    return thresholds

def clean_data_by_angle(gt_anno, dt_anno, current_cls_name, difficulty, difficulty_section):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()        
        valid_class = -1  # 0表示不忽略掉, 1表示特殊类别，-1会忽略掉
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        # dis = np.linalg.norm(gt_anno["location"][i][[0,1]])  # 拿xz来算。计算gt box底层中心点和相机中心点在bev下的距离
        gt_val = gt_anno["rotation_y"][i]

        if (not ((gt_val > difficulty_section[difficulty]) and \
                 (gt_val <= difficulty_section[difficulty + 1]))
            ): 
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        if 0:
            dt_val = np.linalg.norm(dt_anno["location"][i][[0,1]])
            dt_val = dt_anno["rotation_y"][i]

            if not ((dt_val > difficulty_section[difficulty]) and \
                    (dt_val <= difficulty_section[difficulty + 1])):
                ignore = True
            
            if valid_class == 1 and not ignore:
                ignored_dt.append(0)
            elif (valid_class == 0 or (ignore and (valid_class == 1))):
                ignored_dt.append(1)
            else:
                ignored_dt.append(-1)
        else:
            if valid_class == 1 and not ignore:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

def clean_data_by_distance(gt_anno, dt_anno, current_cls_name, difficulty, difficulty_section):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()        
        valid_class = -1  # 0表示不忽略掉, 1表示特殊类别，-1会忽略掉
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        dis = np.linalg.norm(gt_anno["location"][i][[0,1]])  # 拿xy来算。计算gt box底层中心点和相机中心点在bev下的距离
        # if difficulty == 0:
        #     if (dis > difficulty_section[difficulty]):
        #         ignore = True
        if (not ((dis > difficulty_section[difficulty]) and \
                 (dis <= difficulty_section[difficulty + 1]))
            ): 
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])

    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        if 1: # tp/fp/fn
            dis = np.linalg.norm(dt_anno["location"][i][[0,1]])
            if (not ((dis > difficulty_section[difficulty]) and \
                     (dis <= difficulty_section[difficulty + 1]))
            ): 
                ignore = True
            
            if valid_class == 1 and not ignore:
                ignored_dt.append(0)
            elif (valid_class == 0 or (ignore and (valid_class == 1))):
                ignored_dt.append(1)
            else:
                ignored_dt.append(-1)
        else:
            if valid_class == 1 and not ignore:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def clean_data_by_points(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False

        num_point_gt = gt_anno['num_points_in_gt'][i]
        # dis = np.linalg.norm(gt_anno['location'][i][::2])
        if difficulty == 0:
            if num_point_gt > MAX_POINTS[difficulty]:
                ignore = True
        elif ((num_point_gt > MAX_POINTS[difficulty]) or \
              (num_point_gt <= MAX_POINTS[difficulty - 1])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False 
        num_point_dt = dt_anno['num_points_in_gt'][i]
        # dis = np.linalg.norm(gt_anno['location'][i][::2])
        if difficulty == 0:
            if num_point_dt > MAX_POINTS[difficulty]:
                ignore = True
        elif ((num_point_dt > MAX_POINTS[difficulty]) or \
              (num_point_dt <= MAX_POINTS[difficulty - 1])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1        
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

def _prepare_data(gt_annos, dt_annos, current_class, metric_filter_name, difficulty, difficulty_sections):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = eval(metric_filter_name)(gt_annos[i], dt_annos[i], current_class, difficulty, difficulty_sections)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)

def eval_class(gt_annos,
               dt_annos,
               metric_iou,
               config_option,
               current_class,
               metric_eval,
               compute_aos=False,
               num_parts=100,               
              ):
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric_iou, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    min_overlaps = config_option.eval_params[current_class.lower()]["IOU"]
    difficulty_sections  = config_option.eval_params[current_class.lower()]["metric"][metric_eval]
    metric_filter_name = config_option.metric_filter_map[metric_eval]

    difficultys = list(range(len(difficulty_sections)-1))
    num_minoverlap = len(min_overlaps)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    aos = np.zeros([num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    gt_samples_len = []

    for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, metric_filter_name, difficulty, difficulty_sections)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            gt_samples_len.append(np.sum([np.sum(ignored_gts[i]==0) for i in range(len(ignored_gts))]))
            for k, min_overlap in enumerate(min_overlaps):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])    # tp/(tp+fn)
                    precision[l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) # tp/(tp+fp)
                    if compute_aos:
                        aos[l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[l, k, i] = np.max(
                        precision[l, k, i:], axis=-1) # ? 随着分数的增加precision在减小
                    recall[l, k, i] = np.max(recall[l, k, i:], axis=-1) 
                    if compute_aos:
                        aos[l, k, i] = np.max(aos[l, k, i:], axis=-1)
    ret_dict = {
        "gt_samples_len": gt_samples_len,
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict

def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

def do_eval(gt_annos,
            dt_annos,
            config_option,
            current_class,
            eval_metric_name):
    
    ret = eval_class(gt_annos, dt_annos, 1, config_option, current_class, eval_metric_name)
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    ret = eval_class(gt_annos, dt_annos, 2, config_option, current_class, eval_metric_name)
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    
    return mAP_bev_R40, mAP_3d_R40, ret["gt_samples_len"]

def plot_gt_hist(ap_cls_dicts, cfg, eval_metric_name, metric="gt_samples_len", save_dir=None):
    
    color_set = ['b', 'g', 'r', 'c', 'm']    
    for cls in ap_cls_dicts:
        ap_dict = ap_cls_dicts[cls]
        gt_samples_len = ap_dict[metric]
        difficulty_sections = cfg.eval_params[cls]["metric"][eval_metric_name]
        iou_sections = cfg.eval_params[cls]["IOU"]

        plt.figure()
        width = difficulty_sections[2] - difficulty_sections[1]
        plt.bar(difficulty_sections[1:], gt_samples_len, alpha=0.9, width = width, facecolor = 'yellowgreen', edgecolor = 'white', label='one', lw=1)

        plt.title("%s %s mapR40"%(cls,eval_metric_name))
        plt.savefig("%s/%s_%s.png"%(save_dir, eval_metric_name, cls))


def plot_eval_hist(ap_cls_dicts, cfg, eval_metric_name, metric="mAP3d_R40", save_dir=None):

    color_set = ['b', 'g', 'r', 'c', 'm']
    
    for cls in ap_cls_dicts:
        ap_dict = ap_cls_dicts[cls]
        map_r40 = ap_dict[metric]
        difficulty_sections = cfg.eval_params[cls]["metric"][eval_metric_name]
        iou_sections = cfg.eval_params[cls]["IOU"]
        plt.figure()
        legends = []
        for iou in range(map_r40.shape[1]):            
            plt.plot(difficulty_sections[1:], map_r40[:,iou], color=color_set[iou], marker=".")
            legends.append(["iou: {:.2}".format(iou_sections[iou])])
            # x_shift_1 = [x-2.5 for x in sections]
            # x_shift_2 = [x+2.5 for x in sections]
            # plt.bar(x_shift_1, map_r40[cls,:,0], alpha=0.9, width = 5, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)
            # plt.bar(x_shift_2, map_r40[cls,:,1], alpha=0.9, width = 5, facecolor = 'yellowgreen', edgecolor = 'white', label='one', lw=1)

        plt.title("%s %s mapR40"%(cls,eval_metric_name))
        plt.legend(legends)
        plt.grid()
        plt.savefig("%s/%s_%s.png"%(save_dir, eval_metric_name, cls))


def get_metric3D_eval_result(gt_annos, dt_annos, config_option, eval_metric_name):
    
    ret_dict = {}
    class_names = [config_option.class_to_name[k] for k in config_option.class_to_name]

    for cur_class_name in class_names:
        cur_class_name = cur_class_name.lower()
        ret_dict[cur_class_name] = {}
        mAPbev_R40, mAP3d_R40, gt_samples_len = do_eval(
            gt_annos, dt_annos, config_option, cur_class_name, eval_metric_name)

        ret_dict[cur_class_name]["mAPbev_R40"] = mAPbev_R40
        ret_dict[cur_class_name]["mAP3d_R40"]  = mAP3d_R40  
        ret_dict[cur_class_name]["gt_samples_len"]  = gt_samples_len 

        with np.printoptions(precision=3, suppress=True):
            print("**************** %s interval result: ********************"%eval_metric_name)
            print("cur_class_name: ", cur_class_name)
            print("mapbev_R40:\n{}, \nmap3d_R40:\n{}, \ngt_samples_len:\n{}".format(mAPbev_R40,mAP3d_R40, gt_samples_len))
    return ret_dict

def gather_annos(gt_data_dir, pred_dir):
    label_dir = os.path.join(gt_data_dir, "label_2")
    split = "val"
    split_file = os.path.join(os.path.dirname(gt_data_dir), "ImageSets", split + ".txt")

    idx_list = open(split_file).readlines()
    idx_list = [arr.strip() for arr in idx_list]
    gt_img_ids = sorted(idx_list)

    pred_img_ids = os.listdir(pred_dir)
    pred_img_ids = [arr.split('.')[0] for arr in pred_img_ids]
    pred_img_ids = sorted(pred_img_ids)

    assert(len(gt_img_ids)==len(pred_img_ids))
    dt_annos = get_label_annos_str(Path(pred_dir),  pred_img_ids)  #dim: hwl->lhw
    gt_annos = get_label_annos_str(Path(label_dir), gt_img_ids)
    
    return gt_annos, dt_annos

def debug_eval():
    # 坐标系: 前左上
    # 符合KITTI格式的输入要求

    cfg_file = "eval_config.yaml"
    cfg_from_yaml_file(cfg_file, cfg)

    pred_dir = "/data7/zlh/lidar_detection/OpenPCDet/output/untouch_models/centerpoint_union/centerpoint_pandaset_Ins_0122/eval/epoch_80/test/default/final_result/data"
    gt_data_dir = "/data7/zlh/lidar_detection/OpenPCDet/data/pandaset/pandaset_fov/"

    hist_dir = "%s/../"%pred_dir

    gt_annos, dt_annos = gather_annos(gt_data_dir, pred_dir)

    for eval_metric_name in cfg.eval_metric_names:
        hist_map_file = "%s/%s_map.pkl"%(hist_dir, eval_metric_name)
        if not os.path.exists(hist_map_file) or 1: 
            ap_cls_dicts = get_metric3D_eval_result(gt_annos, dt_annos, cfg, eval_metric_name)
            with open(hist_map_file, "wb") as fid:
                pickle.dump(ap_cls_dicts, fid)
        else:
            with open(hist_map_file, "rb") as fid:
                ap_cls_dicts = pickle.load(fid)
            with np.printoptions(precision=3, suppress=True):
                for key in ap_cls_dicts:
                    ap_dict = ap_cls_dicts[key]
                    print("**************** %s [interval*iou] result: ********************"%eval_metric_name)
                    print("cur_class_name: ", key)
                    print("mapbev_R40:\n{}, \nmap3d_R40:\n{}, \ngt_samples_len:\n{}".format(ap_dict["mAPbev_R40"], ap_dict["mAP3d_R40"], ap_dict["gt_samples_len"]))
                    # print("mapbev_R40:\n{}, \nmap3d_R40:\n{}".format(ap_dict["mAPbev_R40"].astype(np.float32), ap_dict["mAP3d_R40"].astype(np.float32)))

        print("pred_dir: ", pred_dir)
        
        hist_dir_3d  = os.path.join(hist_dir, "mAP3d_R40")
        hist_dir_bev = os.path.join(hist_dir, "mAPbev_R40")
        hist_dir_gt = os.path.join(hist_dir, "gt_samples_len")
        
        os.makedirs(hist_dir_3d,  exist_ok=True)
        os.makedirs(hist_dir_bev, exist_ok=True)
        os.makedirs(hist_dir_gt, exist_ok=True)
            
        plot_eval_hist(ap_cls_dicts, cfg, eval_metric_name, metric="mAP3d_R40",  save_dir=hist_dir_3d)
        plot_eval_hist(ap_cls_dicts, cfg, eval_metric_name, metric="mAPbev_R40", save_dir=hist_dir_bev)
        plot_gt_hist(ap_cls_dicts, cfg, eval_metric_name, metric="gt_samples_len", save_dir=hist_dir_gt)

if __name__ == '__main__':
    debug_eval()