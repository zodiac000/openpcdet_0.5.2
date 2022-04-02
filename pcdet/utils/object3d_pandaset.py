import numpy as np


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    """
    主要是这里和untouch不同
    :param cls_type:
    :return:
    """
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


def convert_panda_type(name):
    panda_to_train = {'Car': 'Car',
                      'Medium-sized_Truck': 'Truck',
                      'Pickup_Truck': 'Truck',
                      'Semi-truck': 'Truck',
                      'Towed_Object': 'Car',
                      'Motorcycle': 'Cyclist',
                      'Other_Vehicle_-_Construction_Vehicle': 'Other',
                      'Other_Vehicle_-_Uncommon': 'Other',
                      'Other_Vehicle_-_Pedicab': 'Other',
                      'Emergency_Vehicle': 'Other',
                      'Bus': 'Truck',
                      'Personal_Mobility_Device': 'Cyclist',
                      'Motorized_Scooter': 'Cyclist',
                      'Bicycle': 'Cyclist',
                      'Train': 'Other',
                      'Trolley': 'Other',
                       'Tram_/_Subway': 'Other',
                      'Pedestrian': 'Pedestrian',
                      'Pedestrian_with_Object': 'Pedestrian',
                      'Animals_-_Bird': 'Other',
                      'Animals_-_Other': 'Other',
                      'Pylons': 'Other',
                      'Road_Barriers': 'Other',
                      'Signs': 'Other',
                      'Cones': 'Other',
                      'Construction_Signs': 'Other',
                      'Temporary_Construction_Barriers': 'Other',
                      'Rolling_Containers': 'Other'
                      }
    return panda_to_train[name]

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = convert_panda_type(label[-1])
        self.cls_id = cls_type_to_id(self.cls_type)
        self.x = float(label[0])
        self.y = float(label[1])
        self.z = float(label[2])
        self.l = float(label[3])
        self.w = float(label[4])
        self.h = float(label[5])
        self.ry = float(label[6])
        self.loc = np.array([self.x, self.y, self.z], dtype=np.float32)
        self.dis_to_lidar = np.linalg.norm(self.loc)
        self.score = float(label[7]) if label.__len__() >= 8 else -1.0

        self.level_str = None
        # self.level = self.get_kitti_obj_level()

    # def get_kitti_obj_level(self):
    #     height = float(self.box2d[3]) - float(self.box2d[1]) + 1
    #
    #     if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
    #         self.level_str = 'Easy'
    #         return 0  # Easy
    #     elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
    #         self.level_str = 'Moderate'
    #         return 1  # Moderate
    #     elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
    #         self.level_str = 'Hard'
    #         return 2  # Hard
    #     else:
    #         self.level_str = 'UnKnown'
    #         return -1

    def rotz(self, t):
        """ Rotation about the z-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # def generate_corners3d(self):
    #     """
    #     generate corners3d representation for this object
    #     :return corners_3d: (8, 3) corners of box3d in camera coord
    #     """
    #     # TODO: (du) 确定下这块代码在lidar坐标系下对不对？
    #     l, h, w = self.l, self.h, self.w
    #     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    #     y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    #     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    #
    #     R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
    #                   [0, 1, 0],
    #                   [-np.sin(self.ry), 0, np.cos(self.ry)]])
    #     corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    #     corners3d = np.dot(R, corners3d).T
    #     corners3d = corners3d + self.loc
    #     return corners3d

    def generate_corners3d(self, return_centers=False):
        """
        根据kitti lidar提供的激光上3d box的标签，转换到图像上。
        激光雷达坐标系定义如下：x轴向前，y轴向左，z轴向上。
        0,1为车头方向
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        :param box3d: cx, cy, cz, l, w, h, yaw  激光雷达下的3dbox，xyz是3dbox中心点坐标
        :param cam2velo:
        :param cam_intrinsic:
        :param return_centers:
        :return:
            corners: [8,3]  [1,3]
        """
        # compute rotational matrix around yaw axis
        # 加pi/2是这里保存的是与lidar的x轴夹角，而下面角点默认01车头向右，所以应该先顺时针旋转90度。
        R = self.rotz(self.ry + np.pi / 2)  # 注意激光雷达和相机坐标系的轴不同

        # 3d bounding box dimensions
        l, w, h = self.l, self.h, self.w

        # 3d bounding box corners:先是地面上的4个点注意这里是与激光雷达坐标系相同定义下的局部坐标系
        x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]  # 0,1,2,3,4,5,6,7点
        y_corners = [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2]  # 横着摆放初始目标，目标长在y轴方向上
        z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
        centers = [0, 0, -h / 2]  # 这里要的是底面中心点坐标

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        centers = np.dot(R, np.array(centers).transpose())
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + self.loc[0]
        corners_3d[1, :] = corners_3d[1, :] + self.loc[1]
        corners_3d[2, :] = corners_3d[2, :] + self.loc[2]
        centers[0] += self.loc[0]
        centers[1] += self.loc[1]
        centers[2] += self.loc[2]
        # print('cornsers_3d: ', corners_3d, corners_3d.shape)
        if return_centers:
            return corners_3d.transpose(), centers.transpose()
        else:
            return corners_3d.transpose()

    def to_str(self):
        print_str = '%s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.h, self.w, self.l, self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
