import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval
from .kitti_common import get_label_annos

# for debug
# import sys, os
# sys.path.append(os.path.realpath(__file__))
# print(sys.path)
# from rotate_iou import rotate_iou_gpu_eval  # for test.sh
# from kitti_common import get_label_annos

MAX_DISTANCE = [10, 20, 30, 40, 50, 60, 70]
MAX_POINTS = [200, 500, 2000, 4000, 8000]
distance_threshold = 50  # 只考虑这个距离内的gt和pred

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
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    class_id2name = get_untouch_class()
    current_cls_name = class_id2name[current_class]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    # current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        dis = np.linalg.norm(gt_anno["location"][i][:2])
        if dis > distance_threshold:  # 因为训练网格只设置了前后45米，这里设置成60m以内就行了。
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    num_valid_dt = 0
    for i in range(num_dt):
        if (dt_anno["name"][i] == current_cls_name):
            valid_class = 1
            num_valid_dt += 1
        else:
            valid_class = -1
        ignore = False
        dis = np.linalg.norm(dt_anno["location"][i][:2])
        if dis > distance_threshold:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def clean_data_by_distance(gt_anno, dt_anno, current_class, interval_idx=None):
    """
    对距离分段，统计每段的ap。
    :param gt_anno:
    :param dt_anno:
    :param current_class:
    :param interval_idx:
    :return:
    """
    class_id2name = get_untouch_class()
    current_cls_name = class_id2name[current_class]

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i]
        valid_class = -1  # 0表示不忽略掉, 1表示特殊类别，-1会忽略掉
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        dis = np.linalg.norm(gt_anno["location"][i][:2])  # 计算gt box中心点和lidar中心点的距离
        if interval_idx == 0:
            if(dis > MAX_DISTANCE[interval_idx]):
                ignore = True
        elif dis > MAX_DISTANCE[interval_idx] or dis <= MAX_DISTANCE[interval_idx - 1]:
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    # 统计预测框的情况
    num_valid_dt = 0
    for i in range(num_dt):
        if (dt_anno["name"][i] == current_cls_name):
            valid_class = 1
            num_valid_dt += 1
        else:
            valid_class = -1
        # height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        # if height < MIN_HEIGHT[difficulty]:
        #     ignored_dt.append(1)  # 1是有效的
        # TODO: (du) 对dt也要用距离过滤吗？
        ignore = False
        dis = np.linalg.norm(dt_anno["location"][i][:2])
        if interval_idx == 0:
            if(dis > MAX_DISTANCE[interval_idx]):
                ignore = True
        elif dis > MAX_DISTANCE[interval_idx] or dis <= MAX_DISTANCE[interval_idx - 1]:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_dt.append(0)  # 为0才是有效的
        else:
            ignored_dt.append(-1)  # 为-1会被忽略

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def clean_data_by_points(gt_anno, dt_anno, current_class, interval_idx=None):
    """
    对距离分段，统计每段的ap。
    :param gt_anno:
    :param dt_anno:
    :param current_class:
    :param interval_idx:
    :return:
    """
    class_id2name = get_untouch_class()
    current_cls_name = class_id2name[current_class]

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i]
        valid_class = -1  # 0表示不忽略掉, 1表示特殊类别，-1会忽略掉
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        num_point_gt = gt_anno['num_points_in_gt'][i]  # 计算gt box内的点云数量
        dis = np.linalg.norm(gt_anno["location"][i][:2])
        if interval_idx == 0:
            if num_point_gt > MAX_POINTS[interval_idx] or dis > distance_threshold:
                ignore = True
        elif num_point_gt > MAX_POINTS[interval_idx] or num_point_gt <= MAX_POINTS[interval_idx - 1] or dis > distance_threshold:
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    # 统计预测框的情况
    num_valid_dt = 0
    for i in range(num_dt):
        if (dt_anno["name"][i] == current_cls_name):
            valid_class = 1
            num_valid_dt += 1
            # ignored_dt.append(0)
        else:
            valid_class = -1
            # ignored_dt.append(-1)
        ignore = False
        dis = np.linalg.norm(dt_anno["location"][i][:2])
        if dis > distance_threshold:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_dt.append(0)  # 为0才是有效的
        else:
            ignored_dt.append(-1)  # 为-1会被忽略

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
                # iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                #     boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))
                # 最小的上顶点减去最大的下顶点
                ih = (min(boxes[i, 2] + boxes[i, 5] / 2, qboxes[j, 2] + qboxes[j, 5] / 2) -
                      max(boxes[i, 2] - boxes[i, 5] / 2, qboxes[j, 2] - qboxes[j, 5] / 2))

                if ih > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = ih * rinc[i, j]
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
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)  # lidar: x,y,z,l,w,h,ry
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 6]
    gt_alphas = gt_datas[:, 6]
    # dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

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
                  and (valid_detection == NO_DETECTION)):
                  # and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        # elif ((valid_detection != NO_DETECTION)
        #       and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
        #     assigned_detection[det_idx] = True
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
            if (not (assigned_detection[i] or ignored_det[i] == -1 or ignored_threshold[i])):
            # if (not (assigned_detection[i] or ignored_det[i] == -1
            #          or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        # if metric == 0:
        #     overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
        #     for i in range(dc_bboxes.shape[0]):
        #         for j in range(det_size):
        #             if (assigned_detection[j]):
        #                 continue
        #             if (ignored_det[j] == -1 or ignored_det[j] == 1):
        #                 continue
        #             if (ignored_threshold[j]):
        #                 continue
        #             if overlaps_dt_dc[j, i] > min_overlap:
        #                 assigned_detection[j] = True
        #                 nstuff += 1
        fp -= nstuff
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
                             metric,
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
                dontcare,
                metric,
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
                [a["location"][:, [0, 1]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 1]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 1]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 1]] for a in dt_annos_part], 0)
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
            # print("3d overlap_part: ", overlap_part)
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


def _prepare_data(gt_annos, dt_annos, current_class, difficulty, DIForDIS=True, Points=False):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        # rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        if DIForDIS and not Points:
            rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        elif not DIForDIS:
            rets = clean_data_by_distance(gt_annos[i], dt_annos[i], current_class, difficulty)
        elif Points:
            rets = clean_data_by_points(gt_annos[i], dt_annos[i], current_class, difficulty)
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
        # gt_datas = np.concatenate(
        #     [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        # dt_datas = np.concatenate([
        #     dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
        #     dt_annos[i]["score"][..., np.newaxis]
        # ], 1)
        gt_datas = gt_annos[i]["gt_boxes_lidar"]
        dt_datas = np.concatenate([dt_annos[i]["boxes_lidar"], dt_annos[i]['score'][..., np.newaxis]], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100,
               DIForDIS=True,
               Points=False):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)  # 控制了需要计算几个段
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, DIForDIS=DIForDIS, Points=Points)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, 0, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
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
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
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
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None,
            DIForDIS=True, # 为True不分段，为false分段
            Points=False):  # 为True进行分段，为false则为原始不分段
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = None
    if DIForDIS and not Points:
        difficultys = [0]
    elif not DIForDIS:
        difficultys = list(range(len(MAX_DISTANCE)))
    elif Points:
        difficultys = list(range(len(MAX_POINTS)))
    # difficultys = [0] if DIForDIS else range(len(MAX_DISTANCE))

    mAP_bbox, mAP_bbox_R40 = None, None
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps, DIForDIS=DIForDIS, Points=Points)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps, compute_aos, DIForDIS=DIForDIS, Points=Points)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_0_7 = np.array([[0.5, 0.7, 0.5, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.5, 0.5, 0.25, 0.25, 0.5]])
    # overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
    #                         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
    #                         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    # overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
    #                         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
    #                         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = get_untouch_class()  # untouch dataset index
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['rotation_y'].shape[0] != 0:
            if anno['rotation_y'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    # print("mAP_3d_ids: {}, mAP_3d_R40_dis: {}".format(mAP_3d_dis, mAP_3d_R40_dis))
    print("mapbev: {}, map3d:{}, mapaos:{}, mapbev_R40:{}, map3d_R40:{}, mapAos_R40:{}".format(mAPbev,
                                                                                               mAP3d, mAPaos, mAPbev_R40,
                                                                                               mAP3d_R40, mAPaos_R40))
    # for j, curcls in enumerate(current_classes):
    #     # mAP threshold array: [num_minoverlap, metric, class]
    #     # mAP result: [num_class, num_diff, num_minoverlap]
    #     for i in range(min_overlaps.shape[0]):
    #         result += print_str(
    #             (f"{class_to_name[curcls]} "
    #              "AP@{:.2f}:".format(*min_overlaps[i, :, j])))
    #         result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
    #                              f"{mAPbbox[j, 1, i]:.4f}, "
    #                              f"{mAPbbox[j, 2, i]:.4f}"))
    #         result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
    #                              f"{mAPbev[j, 1, i]:.4f}, "
    #                              f"{mAPbev[j, 2, i]:.4f}"))
    #         result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
    #                              f"{mAP3d[j, 1, i]:.4f}, "
    #                              f"{mAP3d[j, 2, i]:.4f}"))
    #
    #         if compute_aos:
    #             result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
    #                                  f"{mAPaos[j, 1, i]:.2f}, "
    #                                  f"{mAPaos[j, 2, i]:.2f}"))
    #             # if i == 0:
    #                # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
    #                # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
    #                # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]
    #
    #         result += print_str(
    #             (f"{class_to_name[curcls]} "
    #              "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
    #         result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
    #                              f"{mAPbbox_R40[j, 1, i]:.4f}, "
    #                              f"{mAPbbox_R40[j, 2, i]:.4f}"))
    #         result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
    #                              f"{mAPbev_R40[j, 1, i]:.4f}, "
    #                              f"{mAPbev_R40[j, 2, i]:.4f}"))
    #         result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
    #                              f"{mAP3d_R40[j, 1, i]:.4f}, "
    #                              f"{mAP3d_R40[j, 2, i]:.4f}"))
    #         if compute_aos:
    #             result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
    #                                  f"{mAPaos_R40[j, 1, i]:.2f}, "
    #                                  f"{mAPaos_R40[j, 2, i]:.2f}"))
    #             if i == 0:
    #                ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
    #                ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
    #                ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]
    #
    #         if i == 0:
    #             # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
    #             # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
    #             # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
    #             # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
    #             # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
    #             # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
    #             # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
    #             # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
    #             # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]
    #
    #             ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
    #             ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
    #             ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
    #             ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
    #             ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
    #             ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
    #             ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
    #             ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
    #             ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict


def get_distance_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    """
    按距离对depth进行划分，然后统计ap
    :param gt_annos:
    :param dt_annos:
    :param current_classes:
    :param PR_detail_dict:
    :return:
    """
    overlap_0_7 = np.array([[0.5, 0.7, 0.5, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.5, 0.5, 0.25, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 1, 5]
    class_to_name = get_untouch_class()  # untouch dataset index
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['rotation_y'].shape[0] != 0:
            if anno['rotation_y'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, \
    mAPaos_R40 = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos,
                         PR_detail_dict=PR_detail_dict, DIForDIS=False)

    ret_dict = {}
    print("**************** Distance interval result: ********************")
    print("max_distance: {}".format(MAX_DISTANCE))
    print("mapbev: {}, map3d:{}, mapaos:{}, mapbev_R40:{}, map3d_R40:{}, mapAos_R40:{}".format(mAPbev,
                                                                                               mAP3d, mAPaos, mAPbev_R40,
                                                                                               mAP3d_R40, mAPaos_R40))
    for j, curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            if i == 1:
                for depth_idx in range(len(MAX_DISTANCE)):
                    ret_dict['{}_map_3d_R40_{}m'.format(class_to_name[curcls], MAX_DISTANCE[depth_idx])] = mAP3d_R40[j, depth_idx, i]
    return result, ret_dict


def get_points_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    """
    按box内点数对gt进行划分，统计ap
    :param gt_annos:
    :param dt_annos:
    :param current_classes:
    :param PR_detail_dict:
    :return:
    """
    overlap_0_7 = np.array([[0.5, 0.7, 0.5, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.5, 0.5, 0.25, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = get_untouch_class()  # untouch dataset index
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['rotation_y'].shape[0] != 0:
            if anno['rotation_y'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, \
    mAPaos_R40 = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos,
                         PR_detail_dict=PR_detail_dict, DIForDIS=True, Points=True)

    ret_dict = {}
    print("**************** Points interval result: ********************")
    print("max_points: {}".format(MAX_POINTS))
    print("mapbev: {}, map3d:{}, mapaos:{}, mapbev_R40:{}, map3d_R40:{}, mapAos_R40:{}".format(mAPbev,
                                                                                               mAP3d, mAPaos, mAPbev_R40,
                                                                                               mAP3d_R40, mAPaos_R40))
    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result

def get_untouch_class():
    class_to_name = {
        1: 'Car',
        2: 'Pedestrian',
        3: 'Cyclist',
        4: 'Truck',
    }  # untouch dataset index  1_based
    return class_to_name


# ============================== debug ===============
def set_to_zero(annos):
    for anno in annos:
        box = anno['bbox']
        anno['bbox'] = np.zeros_like(box)
    return annos


def camera_to_lidar(annos, dt_flag=False):
    boxes_key = 'boxes_lidar' if dt_flag else 'gt_boxes_lidar'
    for anno in annos:
        loc = anno['location']
        dim = anno['dimensions']

        loc_lidar = loc.copy()
        loc_lidar[:, 0] = loc[:, 2]
        loc_lidar[:, 1] = -loc[:, 0]
        loc_lidar[:, 2] = -(loc[:, 1] - dim[:, 1] / 2)
        dim_lidar = dim.copy()
        dim_lidar[:, 2] = dim[:, 1]
        dim_lidar[:, 1] = dim[:, 2]
        ry = -np.pi/2 - anno['rotation_y']
        anno['location'] = loc_lidar
        anno['dimensions'] = dim_lidar
        anno['rotation_y'] = ry
        anno[boxes_key] = np.concatenate([loc_lidar, dim_lidar, ry[:, None]], axis=1)
    return annos


def debug_eval():
    import os
    from pathlib import Path

    # pred_dir = "/data8/duzhe/code/auto-drive/lidar/point_det/OpenPCDet/tools/work_dirs/debug_eval/final_result/data"
    # gt_data_dir = "/data8/duzhe/dataset/opendata/kitti/object_detect/train_data/training"
    pred_dir = r"D:\tmp-file\untouch\perception\train\pointpillars\debug_eval\final_result\data"
    gt_data_dir = r"D:\code\dataset\opendata\kitti\kitti\training"
    label_dir = os.path.join(gt_data_dir, "label_2")
    split = "val"
    split_file = os.path.join(os.path.dirname(gt_data_dir), "ImageSets", split + ".txt")

    idx_list = [int(x.strip()) for x in open(split_file).readlines()]
    img_ids = sorted(idx_list)
    dt_annos = get_label_annos(Path(pred_dir))  #dim: hwl->lhw
    gt_annos = get_label_annos(Path(label_dir), img_ids)
    # dt_annos = set_to_zero(dt_annos)  # test no 2d box output, all ap will be zero
    # gt_annos = set_to_zero(gt_annos)
    dt_annos = camera_to_lidar(dt_annos, dt_flag=True)
    gt_annos = camera_to_lidar(gt_annos)
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    ap_result_str, ap_dict = get_official_eval_result(gt_annos, dt_annos, class_names)
    ap_result_str, ap_dict = get_distance_eval_result(gt_annos, dt_annos, class_names)
    print(ap_result_str, ap_dict)


if __name__ == '__main__':
    debug_eval()