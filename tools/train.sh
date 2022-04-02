# kitti pp
#CUDA_VISIBLE_DEVICES=2 python train.py --cfg_file cfgs/kitti_models/test_pointpillar.yaml \
#--extra_tag kitti_debug


# mutli-git gpu
#bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/test_pointpillar.yaml \
#--extra_tag kitti_with_intensity

# pandaset
#bash scripts/dist_train.sh 2 --cfg_file cfgs/lidar_pandaset_models/pointpillar_baseline.yaml \
#--extra_tag pandaset_baseline_0125

#bash scripts/dist_train.sh 2 --cfg_file cfgs/lidar_pandaset_models/pointpillar_exp1.yaml \
#--extra_tag pandaset_exp1_0125

#bash scripts/dist_train.sh 2 --cfg_file cfgs/lidar_pandaset_models/pointpillar_exp3.yaml \
#--extra_tag pandaset_exp3_0125

#bash scripts/dist_train.sh 2 --cfg_file cfgs/lidar_pandaset_models/pointpillar_exp2.yaml \
#--extra_tag pandaset_exp2_0127
#CUDA_VISIBLE_DEVICES=4 python train.py --cfg_file cfgs/lidar_pandaset_models/pointpillar_exp2.yaml \
#--extra_tag pandaset_exp2_0127

#bash scripts/dist_train.sh 2 --cfg_file cfgs/lidar_pandaset_models/pointpillar_exp4.yaml \
#--extra_tag pandaset_exp4_0125

#bash scripts/dist_train.sh 2 --cfg_file cfgs/lidar_pandaset_models/pointpillar_exp5.yaml \
#--extra_tag pandaset_exp5_0125

bash scripts/dist_train.sh 4 --cfg_file cfgs/lidar_pandaset_models/pointpillar_multi_head.yaml \
--extra_tag pandaset_exp_multi_head_0315