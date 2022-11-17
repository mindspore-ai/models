"""Configurations."""
import json
from easydict import EasyDict as edict


efficientnet_b1_config_ascend = edict({
    "train_url": None,
    "train_path": None,
    "data_url": None,
    "data_path": None,
    "checkpoint_url": None,
    "checkpoint_path": None,
    "eval_data_url": None,
    "eval_data_path": None,
    "eval_interval": 20,
    "modelarts": False,
    "device_target": "Ascend",
    "run_distribute": False,
    "begin_epoch": 0,
    "end_epoch": 100,
    "total_epoch": 350,

    "dataset": "imagenet",
    "num_classes": 1000,
    "batchsize": 128,

    "lr_scheme": "linear",
    "lr": 0.15,
    "lr_init": 0.0001,
    "lr_end": 5e-5,
    "warmup_epochs": 2,

    "use_label_smooth": True,
    "label_smooth_factor": 0.1,

    "conv_init": "TruncatedNormal",
    "dense_init": "TruncatedNormal",

    "optimizer": "rmsprop",
    "loss_scale": 1024,
    "opt_momentum": 0.9,
    "wd": 1e-5,
    "eps": 0.001,

    "device_num": 1,
    "device_id": 0,

    "model": "efficientnet-b1",
    "input_size": (240, 240),
    "width_coeff": 1.0,
    "depth_coeff": 1.1,
    "dropout_rate": 0.2,
    "drop_connect_rate": 0.2,

    "save_ckpt": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 10
})


def show_config(cfg):
    split_line_up = "==================================================\n"
    split_line_bt = "\n=================================================="
    print(split_line_up,
          json.dumps(cfg, ensure_ascii=False, indent=2),
          split_line_bt, flush=True)


def organize_configuration(cfg, args):
    """Add parameters from command-line into configuration."""
    args_dict = vars(args)
    for item in args_dict.items():
        cfg[item[0]] = item[1]
