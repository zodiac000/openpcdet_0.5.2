# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 14:58
# @Author  : du
# @File    : plot_utils.py
# @desc    : 
# @usage   :
import matplotlib.pyplot as plt
import os


###################################### plot result #########################################
def plot_result(x, y, label_x="x", label_y="y", save_dir="plot.jpg"):
    plt.plot(x, y, '-', label=label_y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.grid()
    plt.xlim(0, x[-1])
    plt.ylim(0)
    plt.tight_layout()

    plt.savefig(save_dir, dpi=200)
    # plt.close()


def plot_result_bar(x, y, label_x="x", label_y="y", save_dir="plot.jpg"):
    plt.bar(x, y, align='center', width=10)
    # plt.plot(x, y, '-', label=label_y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.grid()
    # plt.xlim(0, x[-1])
    plt.ylim(0)
    plt.tight_layout()

    plt.savefig(save_dir, dpi=200)
    # plt.show()

def plot_results(x_lists, y_lists, labels, label_x="x", label_y="y", save_dir="plot.jpg"):
    """
    多个结果绘制在一张图上
    :param x_lists: list，一个元素对应一条曲线
    :param y_lists:
    :param labels: 每条曲线的标签
    :param label_x: x轴的标签
    :param label_y: y轴的标签
    :param save_dir:
    :return:
    """
    for x, y, label in zip(x_lists, y_lists, labels):
        # plt.plot(x, y, '-', label=label)
        plt.bar(x, y, align='center')
        # plt.hist(x, bins=, )
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.grid()
    plt.xlim(0, x_lists[0][-1])
    plt.ylim(0)
    plt.tight_layout()
    plt.legend(loc="upper right")

    plt.savefig(save_dir, dpi=200)
    # plt.close()

def plot_distance_res(ret_dict):
    class_name = "Car"
    # plot_keys = ["image", "bev", "3d"]
    plot_keys = ["3d"]
    ext = "R40"
    x = [10, 20, 30, 40, 50, 60, 70]
    x = [interval - 5 for interval in x]
    one_fig = False   # 结果画在同一张图上
    y_lists = []
    for plot_key in plot_keys:
        y = []
        for key, value in ret_dict.items():
            if plot_key in key and class_name in key and ext in key:
                print(key, value)
                y.append(value)
        y_lists.append(y)
        if not one_fig:
            plot_result_bar(x, y, label_x="depth", label_y=class_name + "_AP_40",
                        save_dir=os.path.join("/data8/duzhe/code/auto-drive/lidar/point_det/OpenPCDet/tools/work_dirs/plot_res", plot_key + ".jpg"))
    if one_fig:
        plot_results([x, x, x], y_lists, plot_keys, label_x="depth", label_y="AP_40",
                     save_dir=os.path.join("/data7/zlh/lidar_detection/OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python", "train_kitti_75_ap40.jpg"))

def plot_heading_res(ret_dict):
    class_name = "Car"
    plot_keys = ["image", "bev", "3d"]
    ext = "R40"
    x = [0.26,0.52,0.78]
    one_fig = True   # 结果画在同一张图上
    y_lists = []
    for plot_key in plot_keys:
        y = []
        for key, value in ret_dict.items():
            if plot_key in key and class_name in key and ext in key:
                print(key, value)
                y.append(value)
        y_lists.append(y)
        if not one_fig:
            plot_result(x, y, label_x="heading", label_y="AP_40",
                        save_dir=os.path.join("/data7/zlh/lidar_detection/OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python", plot_key + ".jpg"))
    if one_fig:
        plot_results([x, x, x], y_lists, plot_keys, label_x="depth", label_y="AP_40",
                     save_dir=os.path.join("/data7/zlh/lidar_detection/OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python", "train_kitti_75_ap40.jpg"))
#########################################################################################


def debug_plt():
    x = [10, 20, 30, 40, 50, 60, 70]
    x = [interval - 5 for interval in x]

    y = [99.43838572097677, 93.53229567204441, 84.50532247949202, 63.57450979398557, 31.347345218356647, 1.4828933085505591, 0]
    plot_result_bar(x, y)

if __name__ == '__main__':
    debug_plt()