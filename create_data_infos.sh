# kitti
#python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

# pandaset
python -m pcdet.datasets.lidar_pandaset_dataset.lidar_pandaset_dataset \
create_lidar_pandaset_infos tools/cfgs/dataset_configs/lidar_pandaset_dataset.yaml
