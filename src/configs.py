from easydict import EasyDict as edict
import argparse
import torch.cuda
import os
import numpy as np
import yaml
from torch import nn

with open("data/data_cfg.yaml", 'r') as file:
    configs = yaml.safe_load(file)

CameraType = edict()
CameraType.realsense_d435i = 0
CameraType.realsense_l515 = 1


Camera_cfg = edict()
Camera_cfg.camera_type = 0  # 0: realsense d435; 1: realsense l515
Camera_cfg.realsense_d435i = edict()
Camera_cfg.realsense_d435i.image_size = (1280, 720)
Camera_cfg.realsense_d435i.fx = 920.451904296875
Camera_cfg.realsense_d435i.fy = 920.516357421875
Camera_cfg.realsense_d435i.cx = 631.1873779296875
Camera_cfg.realsense_d435i.cy = 370.04132080078125
Camera_cfg.realsense_d435i.k1 = 0
Camera_cfg.realsense_d435i.k2 = 0
Camera_cfg.realsense_d435i.p1 = 0
Camera_cfg.realsense_d435i.p2 = 0
Camera_cfg.realsense_d435i.k3 = 0
Camera_cfg.realsense_d435i.camera_height = 0.35  # unit m
Camera_cfg.realsense_d435i.camera_x_offset = 0  # unit m, distance between the origins of Lidar and camera

Camera_cfg.realsense_l515 = edict()
Camera_cfg.realsense_l515.image_size = (480, 640)
Camera_cfg.realsense_l515.fx = 607.175048828125
Camera_cfg.realsense_l515.fy = 607.222900390625
Camera_cfg.realsense_l515.cx = 248.86021423339844
Camera_cfg.realsense_l515.cy = 322.55340576171875
Camera_cfg.realsense_l515.k1 = 0
Camera_cfg.realsense_l515.k2 = 0
Camera_cfg.realsense_l515.p1 = 0
Camera_cfg.realsense_l515.p2 = 0
Camera_cfg.realsense_l515.k3 = 0
Camera_cfg.realsense_l515.camera_height = 0.9  # unit m
Camera_cfg.realsense_l515.camera_x_offset = 0  # unit m, distance between the origins of Lidar and camera
Camera_cfg.realsense_l515.camera_y_offset = 0  # unit m, distance between the origins of Lidar and camera

LidarMode = edict()
LidarMode.image = 0
LidarMode.ptcnn = 1
LidarMode.kpconv = 2

RNNType = edict()
RNNType.gru = 0
RNNType.lstm = 1

ModelType = edict()
ModelType.cvae = 0
ModelType.dlow = 1
ModelType.dlowae = 2
ModelType.terrapn = 3

DistanceFunction = edict()
DistanceFunction.euclidean = 0
DistanceFunction.point_wise = 1

LossDisType = edict()
LossDisType.dtw = 0
LossDisType.hausdorff = 1

Hausdorff = edict()
Hausdorff.average = 0
Hausdorff.max = 1

CollisionLossType = edict()
CollisionLossType.global_dis = 0
CollisionLossType.local_dis = 1

DiversityType = edict()
DiversityType.target_diversity = 0
DiversityType.self_diversity = 1

GTType = edict()
GTType.generated = 0
GTType.demonstration = 1

ActivateFunction = edict()
ActivateFunction.soft = 0
ActivateFunction.tanh = 1

DataName = edict()
DataName.camera = "camera"
DataName.lidar = "lidar"
DataName.lidar2d = "lidar_array"
DataName.vel = "vel"
DataName.imu = "imu"
DataName.path = "path"
DataName.last_poses = "last_poses"
DataName.mu = "mu"
DataName.logvar = "logvar"
DataName.A = "A"
DataName.b = "b"
DataName.y_hat = "y_hat"
DataName.scores = "scores"
DataName.png = "png"
DataName.scan = "scan"
DataName.local_map = "local_map"
DataName.all_paths = "all_paths"
DataName.pose = "pose"

LossDictKeys = edict()
LossDictKeys.loss = "loss"
LossDictKeys.dlow_kld_loss = "dlow_kld_loss"
LossDictKeys.vae_kld_loss = "vae_kld_loss"
LossDictKeys.last_point_loss = "last_point_loss"
LossDictKeys.distance_loss = "distance_loss"
LossDictKeys.diversity_loss = "diversity_loss"
LossDictKeys.collision_loss_max = "collision_loss_max"
LossDictKeys.collision_loss_mean = "collision_loss_mean"
LossDictKeys.coverage_distance = "coverage_distance"
LossDictKeys.coverage_last = "coverage_last"
LossDictKeys.coverage_diverse = "coverage_diverse"
LossDictKeys.asymmetric_loss = "asymmetric_loss"

cfg = edict()
cfg.name = ""
cfg.device = "cuda:0"
cfg.eval = False
cfg.load_snapshot = ""

cfg.data = edict()
cfg.data.file = ""
cfg.data.name = ""
cfg.data.batch_size = 16
cfg.data.num_workers = 8
cfg.data.shuffle = True
cfg.data.training_data_percentage = 0.95
cfg.data.lidar_max_points = 5120
cfg.data.lidar_downsample_vx_size = 0.08
cfg.data.lidar_mode = LidarMode.image
cfg.data.lidar_threshold = configs["rosbag"]["lidar"]["threshold"]
cfg.data.vel_num = 10
cfg.data.use_local_map = False
cfg.data.local_map_threshold = int(configs["local_map"]["target_distance"] * 2 / configs["local_map"]["resolution"])
cfg.data.w_eval = False

cfg.evaluation = edict()
cfg.evaluation.max_epoch = 400
cfg.evaluation.max_iteration_per_epoch = 5000
cfg.evaluation.display = True
cfg.evaluation.root = ""
cfg.evaluation.local_map_resolution = configs["local_map"]["resolution"]
cfg.evaluation.local_map_threshold = cfg.data.local_map_threshold

cfg.training = edict()
cfg.training.no_eval = False
cfg.training.max_epoch = 500
cfg.training.max_iteration_per_epoch = 5000
cfg.training.lr = 1e-4
cfg.training.lr_decay = 0.5
cfg.training.lr_decay_steps = 4
cfg.training.weight_decay = 1e-6
cfg.training.grad_acc_steps = 5

cfg.loss_eval = edict()
cfg.loss_eval.type = ModelType.cvae
# cfg.loss_eval.gt_type = GTType.generated
cfg.loss_eval.distance_type = LossDisType.hausdorff
cfg.loss_eval.hausdorff_dis = Hausdorff.average
cfg.loss_eval.dtw_use_cuda = True
cfg.loss_eval.dtw_gamma = 0.1
cfg.loss_eval.dtw_normalize = True
cfg.loss_eval.dtw_dist_func = DistanceFunction.euclidean
cfg.loss_eval.scale_waypoints = 1.0
cfg.loss_eval.dlow_sigma = 100.0
cfg.loss_eval.local_map_resolution = configs["local_map"]["resolution"]
cfg.loss_eval.collision_threshold = 1.0 / configs["local_map"]["resolution"]

cfg.loss_eval.collision_type = CollisionLossType.global_dis
cfg.loss_eval.collision_detection_dis = int(1.0 / configs["local_map"]["resolution"])
cfg.loss_eval.local_map_sample_dis = 2
cfg.loss_eval.local_map_threshold = cfg.data.local_map_threshold
cfg.loss_eval.diversity_type = DiversityType.self_diversity

cfg.loss_eval.asymmetric_ratio = 0
cfg.loss_eval.last_ratio = 2.0
cfg.loss_eval.vae_kld_ratio = 1.0
cfg.loss_eval.dlow_kld_ratio = 0.01
cfg.loss_eval.distance_ratio = 10.0
cfg.loss_eval.diversity_ratio = 1000.0
cfg.loss_eval.collision_mean_ratio = 10.0
cfg.loss_eval.collision_max_ratio = 10.0

cfg.loss_eval.coverage_with_last = True
cfg.loss_eval.coverage_distance_ratio = 10
cfg.loss_eval.coverage_last_ratio = 1
cfg.loss_eval.coverage_diverse_ratio = 1

cfg.logger = edict()
cfg.logger.log_steps = 5
cfg.logger.verbose = False
cfg.logger.reset_num_timesteps = True
cfg.logger.log_name = "./loginfo/"

cfg.model = edict()
cfg.model.perception = edict()
cfg.model.perception.fix_perception = False
cfg.model.perception.vel_dim = 20
cfg.model.perception.vel_out = 256
cfg.model.perception.lidar_out = 512
cfg.model.perception.lidar_norm_layer = False
cfg.model.perception.lidar_num = configs["rosbag"]["lidar"]["number"]
cfg.model.perception.lidar_mode = cfg.data.lidar_mode
cfg.model.perception.kpconv = edict()

cfg.model.perception.use_local_map = False
cfg.model.perception.local_map_size = cfg.data.local_map_threshold * 2
cfg.model.perception.local_map_out = cfg.model.perception.lidar_out + cfg.model.perception.vel_out

cfg.model.dlow = edict()
cfg.model.dlow.w_others = False
cfg.model.dlow.transformer_heads = 4
cfg.model.dlow.activation_func = None
cfg.model.dlow.fix_cvae = False
cfg.model.dlow.model_type = ModelType.cvae
cfg.model.dlow.rnn_type = RNNType.gru
cfg.model.dlow.perception_in = cfg.model.perception.lidar_out + cfg.model.perception.vel_out
cfg.model.dlow.vae_zd = 512
cfg.model.dlow.vae_output_threshold = 1
cfg.model.dlow.paths_num = 5
cfg.model.dlow.waypoints_num = 20
cfg.model.dlow.waypoint_dim = 2
cfg.model.dlow.fix_first = False
cfg.model.dlow.cvae_file = None

cfg.experiment = edict()
cfg.experiment.data_file = ""
cfg.experiment.saving_root = ""
cfg.experiment.name = "experiment"
cfg.experiment.display = True
cfg.experiment.cropping_row = int(Camera_cfg.realsense_d435i.image_size[1] / 2.0)
cfg.experiment.metrics = edict()
cfg.experiment.metrics.hausdorff_dis = Hausdorff.max
cfg.experiment.metrics.local_map_resolution = cfg.loss_eval.local_map_resolution
cfg.experiment.metrics.root = "experiments/results"
cfg.experiment.metrics.camera_type = 0

cfg.experiment.data = edict()
cfg.experiment.data.root = "datasets/local_map_files_120"
cfg.experiment.data.idx = 1034
cfg.experiment.data.local_map_threshold = cfg.data.local_map_threshold
cfg.experiment.data.vel_num = cfg.data.vel_num
cfg.experiment.data.batch_size = 8

cfg.experiment.rosbag = edict()
cfg.experiment.rosbag.root = "experiments/bags"
cfg.experiment.rosbag.name = ""

cfg.experiment.rosbag.lidar_channnels = 16
cfg.experiment.rosbag.lidar_angle_range = 200 * np.pi / 180.0
cfg.experiment.rosbag.lidar_threshold = 100
cfg.experiment.rosbag.lidar_horizons = 1824

cfg.experiment.rosbag.lidar_num = 3
cfg.experiment.rosbag.scan_num = 1
cfg.experiment.rosbag.vel_num = cfg.data.vel_num
cfg.experiment.rosbag.camera_num = 1


def get_args():
    parser = argparse.ArgumentParser(description='mapping')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--name', type=str, default="new")

    # Data settings
    parser.add_argument('--data_root', type=str, default="data/local_map_files_120")
    parser.add_argument('--data_name', type=str, default="data.pkl")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--not_shuffle', action='store_true', default=False)
    parser.add_argument('--training_percentage', type=float, default=0.95)
    parser.add_argument('--lidar_mode', type=int, default=LidarMode.image, help="0: lidar image;  1: lidar point cloud")

    # Training settings
    parser.add_argument('--max_epoch', type=int, default=1000, help="max epochs")
    parser.add_argument('--lr_decay_steps', type=int, default=2, help="number of waypoints")
    parser.add_argument('--grad_step', type=int, default=1, help="number of waypoints")
    parser.add_argument('--gamma', type=float, default=0.95, help="number of waypoints")

    parser.add_argument('--snap_shot', type=str, default="")
    parser.add_argument('--dlow_type', type=int, default=ModelType.dlowae, help="0: CVAE, 1: dlow, 2: dlowae")
    parser.add_argument('--rnn', type=int, default=RNNType.gru, help="0: gru;  1: lstm")
    parser.add_argument('--fix_cvae', action='store_true', default=False)
    parser.add_argument('--w_eval', action='store_true', default=False)

    # Models settings
    parser.add_argument('--norm_lidar', action='store_true', default=False)
    parser.add_argument('--fix_obs', action='store_true', default=False)
    parser.add_argument('--use_local_map', action='store_true', default=False)
    parser.add_argument('--waypoints_num', type=int, default=16, help="number of waypoints")
    parser.add_argument('--paths_num', type=int, default=5, help="number of paths of dlow")
    parser.add_argument('--activation_func', type=int, default=None, help="0 softsign,  1 tanh")
    parser.add_argument('--w_others', action='store_true', default=False)

    # Loss settings
    parser.add_argument('--distance_type', type=int, default=LossDisType.hausdorff, help="0: dtw;  1: hausdorff")
    parser.add_argument('--collision_type', type=int, default=CollisionLossType.local_dis, help="number of waypoints")
    # parser.add_argument('--gt_type', type=int, default=GTType.generated, help="0: generated, 1: demonstration")
    parser.add_argument('--vae_kld_ratio', type=float, default=100, help="number of waypoints")
    parser.add_argument('--dlow_kld_ratio', type=float, default=0.01, help="number of waypoints")
    parser.add_argument('--last_ratio', type=float, default=0, help="number of waypoints")
    parser.add_argument('--distance_ratio', type=float, default=0, help="number of waypoints")
    parser.add_argument('--diversity_ratio', type=float, default=0, help="number of waypoints")
    parser.add_argument('--collision_mean_ratio', type=float, default=100000, help="number of waypoints")
    parser.add_argument('--collision_max_ratio', type=float, default=100, help="number of waypoints")
    parser.add_argument('--coverage_distance_ratio', type=float, default=100, help="number of waypoints")
    parser.add_argument('--coverage_last_ratio', type=float, default=10, help="number of waypoints")
    parser.add_argument('--asymmetric_ratio', type=float, default=0, help="number of waypoints")
    parser.add_argument('--coverage_diverse_ratio', type=float, default=10, help="number of waypoints")

    return parser.parse_args()


def get_configs():
    args = get_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        torch.cuda.set_device(args.device)
        cfg.device = "cuda:" + str(args.device)
        cfg.loss_eval.dtw_use_cuda = True
    else:
        cfg.device = "cpu"
        cfg.loss_eval.dtw_use_cuda = False

    cfg.eval = args.eval
    cfg.load_snapshot = args.snap_shot

    cfg.training.lr_decay_steps = args.lr_decay_steps
    cfg.training.lr_decay = args.gamma
    cfg.training.weight_decay = 1e-6
    cfg.training.grad_acc_steps = args.grad_step
    cfg.training.max_epoch = args.max_epoch
    cfg.data.w_eval = cfg.training.w_eval = args.w_eval

    cfg.evaluation.root = args.data_root
    cfg.data.file = args.data_root
    cfg.data.name = args.data_name
    cfg.data.num_workers = args.workers
    cfg.data.batch_size = args.batch_size
    cfg.data.shuffle = ~args.not_shuffle
    cfg.data.training_data_percentage = args.training_percentage
    cfg.data.lidar_mode = cfg.model.perception.lidar_mode = args.lidar_mode

    cfg.model.perception.lidar_norm_layer = args.norm_lidar
    if not cfg.model.perception.lidar_norm_layer:
        cfg.data.lidar_threshold = None

    if args.activation_func == ActivateFunction.tanh:
        cfg.model.dlow.activation_func = nn.Tanh
    elif args.activation_func == ActivateFunction.soft:
        cfg.model.dlow.activation_func = nn.Softsign
    else:
        cfg.model.dlow.activation_func = None
    cfg.model.dlow.model_type = cfg.loss_eval.type = args.dlow_type
    cfg.model.dlow.rnn_type = args.rnn
    cfg.model.dlow.fix_cvae = args.fix_cvae
    cfg.model.dlow.waypoints_num = args.waypoints_num
    cfg.model.dlow.paths_num = args.paths_num
    cfg.model.dlow.w_others = args.w_others
    cfg.model.perception.fix_perception = args.fix_obs
    cfg.data.use_local_map = cfg.model.perception.use_local_map = args.use_local_map

    # cfg.loss_eval.gt_type = args.gt_type
    cfg.loss_eval.asymmetric_ratio = args.asymmetric_ratio
    cfg.loss_eval.distance_type = args.distance_type
    cfg.loss_eval.collision_type = args.collision_type
    cfg.loss_eval.collision_mean_ratio = args.collision_mean_ratio
    cfg.loss_eval.collision_max_ratio = args.collision_max_ratio
    cfg.loss_eval.last_ratio = args.last_ratio
    cfg.loss_eval.vae_kld_ratio = args.vae_kld_ratio
    cfg.loss_eval.dlow_kld_ratio = args.dlow_kld_ratio
    cfg.loss_eval.distance_ratio = args.distance_ratio
    cfg.loss_eval.diversity_ratio = args.diversity_ratio
    cfg.loss_eval.coverage_distance_ratio = args.coverage_distance_ratio
    cfg.loss_eval.coverage_last_ratio = args.coverage_last_ratio
    cfg.loss_eval.coverage_diverse_ratio = args.coverage_diverse_ratio

    if args.name is not None:
        cfg.name = args.name

    cfg.name += "_wn{}".format(args.waypoints_num)
    cfg.name += "_pn{}".format(args.paths_num)
    cfg.name += "_lm{}".format(args.lidar_mode)
    cfg.name += "_T{}".format(args.dlow_type)
    # cfg.name += "_gt{}".format(args.gt_type)
    cfg.name += "_oth{}".format(args.w_others)
    cfg.name += "_lds{}".format(args.lr_decay_steps)
    cfg.name += "_gs{}".format(args.grad_step)
    cfg.name += "_vkl{}".format(args.vae_kld_ratio)
    cfg.name += "_dkl{}".format(args.dlow_kld_ratio)
    cfg.name += "_lr{}".format(args.last_ratio)
    cfg.name += "_disr{}".format(args.distance_ratio)
    cfg.name += "_divr{}".format(args.diversity_ratio)
    cfg.name += "_car{}".format(args.collision_mean_ratio)
    cfg.name += "_cmr{}".format(args.collision_max_ratio)
    cfg.name += "_cvlr{}".format(args.coverage_last_ratio)
    cfg.name += "_cvdr{}".format(args.coverage_distance_ratio)
    cfg.name += "_asmr{}".format(args.asymmetric_ratio)
    cfg.name += "_cvdr{}".format(args.coverage_diverse_ratio)
    return cfg
