#### 说明
修改自`kitti-object-eval-python`, 支持lidar坐标系下的3d box的评价指标计算。

- lidar坐标系下的指标计算，调用`eval.py`。

    计算的指标包括3部分。均是对一定距离范围的gt和预测来计算的，因为训练时网络只考虑了中间一部分区域。

    - 原始ap计算，所以gt和det都限制在60米以内。
    - depth计算是划分成多段区间去统计ap的，所以每个都划分到一定距离里了。
    - 根据gt box内点数量划分区间来统计ap，这里也将gt和det限制在60米以内。

