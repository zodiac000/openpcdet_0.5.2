import copy
import pickle

import numpy as np
from skimage import io
import os
from pathlib import Path

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_lidar
from ..dataset import DatasetTemplate


class LidarDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.root_paths = []
        if root_path is None and isinstance(self.dataset_cfg.DATA_PATH, list):
            for root_p in self.dataset_cfg.DATA_PATH:
                self.root_paths.append(root_p)
        else:
            self.root_path = root_path
            self.root_paths.append(str(self.root_path))
        print("root_paths: {}".format(self.root_paths))
        split = 'train' if self.split != 'test' else 'test'
        self.set_split2(split)
        print("after split: ", self.root_path)
        print("after split, data_paths: ", self.root_paths)

        self.lidar_infos = []
        self.include_untouch_data(self.mode)
        self.pass_root_path = root_path

    def set_split2(self, split):
        """
        不调基类的__init__，和set_split做区分。因为训练时如果再调用__init__，会调两遍，数据会double
        :param split:
        :return:
        """
        print("calling set_split2: {}".format(split))
        self.split = split
        split_flag = None
        if self.split == 'train':
            split_flag = 'training'
        else:
            split_flag = self.split
        # print(split, split_flag)

        self.root_split_paths = []
        self.labels_dir = []
        self.lidar_dir = []
        self.sample_id_list = []
        self.name_dir_index_dict = {}
        # print(self.root_paths)
        if len(self.root_paths) > 1:
            for root_idx, root_p in enumerate(self.root_paths):
                self.root_split_paths.append(os.path.join(root_p, split_flag))
                label_dir = os.path.join(root_p, split_flag, 'label')
                label_files = os.listdir(label_dir)
                self.labels_dir.append(label_dir)
                self.lidar_dir.append(os.path.join(root_p, split_flag, 'velodyne'))

                sample_idx_list = []
                for label_file in label_files:
                    file_name = label_file.split(".txt")[0]
                    # print(file_name)
                    self.name_dir_index_dict[file_name] = root_idx
                    sample_idx_list.append(file_name)
                self.sample_id_list.append(sample_idx_list)  # TODO: (du) 需要sorted嗎
        else:
            self.root_split_paths.append(os.path.join(self.root_paths[0], split_flag))
            label_dir = os.path.join(self.root_paths[0], split_flag, 'label')
            self.labels_dir.append(label_dir)
            label_files = os.listdir(label_dir)
            self.lidar_dir.append(os.path.join(self.root_paths[0], split_flag, 'velodyne'))

            sample_idx_list = []
            for label_file in label_files:
                file_name = label_file.split(".txt")[0]
                self.name_dir_index_dict[file_name] = 0
                sample_idx_list.append(file_name)
            self.sample_id_list.append(sample_idx_list)

    def include_untouch_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading PandaSet dataset')
        untouch_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:

            for root_p in self.root_paths:
                infos_path = os.path.join(root_p, str(info_path))
                if not os.path.exists(infos_path):
                    continue
                with open(infos_path, 'rb') as f:
                    infos = pickle.load(f)
                    untouch_infos.extend(infos)

        self.untouch_infos.extend(untouch_infos)

        if self.logger is not None:
            self.logger.info('Total samples for UNTOUCH dataset: %d' % (len(untouch_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.pass_root_path, logger=self.logger
        )
        print("calling set_split2: {}".format(split))
        self.split = split
        split_flag = None
        if self.split == 'train':
            split_flag = 'training'
        else:
            split_flag = self.split
        # print(split, split_flag)

        self.root_split_paths = []
        self.labels_dir = []
        self.lidar_dir = []
        self.sample_id_list = []
        self.name_dir_index_dict = {}
        # print(self.root_paths)
        if len(self.root_paths) > 1:
            for root_idx, root_p in enumerate(self.root_paths):
                self.root_split_paths.append(os.path.join(root_p, split_flag))
                label_dir = os.path.join(root_p, split_flag, 'label')
                label_files = os.listdir(label_dir)
                self.labels_dir.append(label_dir)
                self.lidar_dir.append(os.path.join(root_p, split_flag, 'velodyne'))

                sample_idx_list = []
                for label_file in label_files:
                    file_name = label_file.split(".txt")[0]
                    self.name_dir_index_dict[file_name] = root_idx
                    sample_idx_list.append(file_name)
                self.sample_id_list.append(sample_idx_list)  # TODO: (du) 需要sorted嗎
        else:
            self.root_split_paths.append(os.path.join(self.root_paths[0], split_flag))
            label_dir = os.path.join(self.root_paths[0], split_flag, 'label')
            self.labels_dir.append(label_dir)
            label_files = os.listdir(label_dir)
            self.lidar_dir.append(os.path.join(self.root_paths[0], split_flag, 'velodyne'))

            sample_idx_list = []
            for label_file in label_files:
                file_name = label_file.split(".txt")[0]
                self.name_dir_index_dict[file_name] = 0
                sample_idx_list.append(file_name)
            self.sample_id_list.append(sample_idx_list)

    def get_lidar(self, list_index, sample_name):
        lidar_file = os.path.join(self.root_split_paths[list_index], 'velodyne', ('%s.bin' % sample_name))
        assert os.path.exists(lidar_file)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self, list_index, sample_name):
        label_file = os.path.join(self.root_split_paths[list_index], 'label', ("%s.txt" % sample_name))
        assert os.path.exists(label_file)
        return object3d_pandaset.get_objects_from_label(label_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_name):
            dir_index = self.name_dir_index_dict[sample_name]
            # print('%s list_index: %s, sample_idx: %s' % (self.split, dir_index, sample_name))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_name}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(dir_index, sample_name)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])  # pandaset中name经过了转换
                # annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                # annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                # annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([0 for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # print("loc: {}, dims: {}, rots: {}".format(loc.shape, dims.shape, rots.shape))
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                annotations['frame_id'] = sample_name

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(dir_index, sample_name)
                    # calib = self.get_calib(sample_name)
                    # pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            for list_idx, sample_id_l in enumerate(sample_id_list):
                # list_idx = [list_idx] * len(sample_id_l)
                infos = executor.map(process_single_scene, sample_id_l)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = os.path.join(self.root_path, 'gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = os.path.join(self.root_path, 'untouch_dbinfos_%s.pkl' % split)
        print("database_save_path: ", database_save_path)
        print("db_info_save_path: ", db_info_save_path)
        os.makedirs(database_save_path, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_name = info['point_cloud']['lidar_idx']
            dir_index = self.name_dir_index_dict[sample_name]
            points = self.get_lidar(dir_index, sample_name)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            # bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_name, names[i], i)
                filepath = os.path.join(database_save_path, filename)
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:   # 这是写入裁剪出的目标点云到硬盘
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    dir_lists = database_save_path.split(os.sep)
                    gt_database = dir_lists[-1]
                    dataset_dir = dir_lists[-2]
                    db_path = os.path.join(dataset_dir, gt_database, filename)  # dataset_dir/gt_database/xxx
                    # print("gt_database: {}, db_path: {}".format(gt_database, dataset_dir))
                    # print("db_path: ", db_path)
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_name, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i],'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
            # break
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:  # 这是写入dbinfos.pkl
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def labels_to_untouch_names(pred_labels):
            # untouch to id. 由config中的顺序决定
            class_to_name = {
                0 : 'Pedestrian',
                1 : 'Car',
                2 : 'Pedestrian',
                3 : 'Cyclist',
                # 3: 'Van',
                # 4: 'Person_sitting',
                4 : 'Truck'
            }
            labels_res = []
            for label in pred_labels:
                labels_res.append(class_to_name[label])
            return np.array(labels_res)

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['name'] = labels_to_untouch_names(pred_labels)
            # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    # bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']   # lwh

                    for idx in range(len(loc)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                 # bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def lidar_to_camera(self, annos):
        """
        将lidar下的结果转为camera坐标系下，用于计算指标
        :param annos:
        :return:
        """
        pass

    def evaluation(self, det_annos, class_names, **kwargs):
        """

        :param det_annos:
        :param class_names: 由config中的顺序决定
        :param kwargs:
        :return:
        """
        if 'annos' not in self.untouch_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval
        from .kitti_object_eval_python import plot_utils

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.untouch_infos]
        # eval_det_annos, eval_gt_annos = self.debug_multi_annos()
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        ap_result_str, ap_dict = kitti_eval.get_distance_eval_result(eval_gt_annos, eval_det_annos, class_names)
        plot_utils.plot_distance_res(ap_dict)
        ap_result_str, ap_dict = kitti_eval.get_points_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.untouch_infos) * self.total_epochs

        return len(self.untouch_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.untouch_infos)

        info = copy.deepcopy(self.untouch_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        dir_index = self.name_dir_index_dict[sample_idx]
        points = self.get_lidar(dir_index, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'dir_index': dir_index,
            # 'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=-1)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            # road_plane = self.get_road_plane(sample_idx)
            # if road_plane is not None:
            #     input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_lidar_infos(dataset_cfg, class_names, data_path, save_path, workers=4, split='train'):
    # training = True if split == 'train' else False
    dataset = LidarDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    # train_split, val_split = 'train', 'val'

    train_filename = os.path.join(save_path, ('untouch_infos_%s.pkl' % split))

    print('---------------Start to generate data infos---------------')
    print("split: ", split)
    dataset.set_split(split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('untouch info {} file is saved to %{}'.format(split, train_filename))

    print('---------------Start create groundtruth database for data augmentation---------------')
    # dataset.set_split(split)
    if split == "train":
        dataset.create_groundtruth_database(train_filename, split=split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    print("sys.argv: ", sys.argv)
    if sys.argv.__len__() > 1 and sys.argv[2] == 'create_untouch_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[4])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        print("Please make sure modify your untouch dataset path in pcdet/datasets/untouch/untouch_dataset.py ! ")
        # 每次修改这里，对新加入的数据生成pickle文件和gt_database.
        Data_DIR = Path("/data8/duzhe/dataset/untouch/lidar_datasets/pandaset")
        create_lidar_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck'],
            data_path=Data_DIR,
            # data_path=Data_DIR / 'data' / 'kitti',
            save_path=Data_DIR,
            split="train"
            # split="test"
        )
    print("finished")
