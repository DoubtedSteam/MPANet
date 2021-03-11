from yacs.config import CfgNode

dataset_cfg = CfgNode()

# config for dataset
dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = "../dataset/SYSU-MM01"

dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = "../dataset/RegDB"

dataset_cfg.market = CfgNode()
dataset_cfg.market.num_id = 751
dataset_cfg.market.num_cam = 6
dataset_cfg.market.data_root = "../dataset/market"
