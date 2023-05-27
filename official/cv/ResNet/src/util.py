import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from src.callback import EvalCallBack
from src.resnet import conv_variance_scaling_initializer


def filter_checkpoint_parameter_by_list(origin_dict, param_filter, cfg):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                cfg.logger.info("Delete parameter from checkpoint: %s", key)
                del origin_dict[key]
                break


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def init_group_params(net, cfg):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': cfg.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


def eval_callback(model, cfg, eval_dataset):
    eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=cfg.eval_interval,
                           eval_start_epoch=cfg.eval_start_epoch, rank_id=cfg.rank_id,
                           save_best_ckpt=cfg.save_best_ckpt, ckpt_directory=cfg.save_ckpt_dir,
                           best_ckpt_name="best_acc.ckpt", metrics_name="acc", logger=cfg.logger)
    return eval_cb


def set_output_dir(cfg):
    """set save ckpt dir"""
    cfg.output_dir = os.path.realpath(os.path.join(cfg.output_dir, cfg.net_name, cfg.dataset))
    cfg.save_ckpt_dir = os.path.join(cfg.output_dir, 'ckpt')
    cfg.log_dir = os.path.join(cfg.output_dir, 'log')
    return cfg


def set_golden_output_dir(cfg):
    """set save ckpt dir"""
    cfg.output_dir = os.path.realpath(os.path.join(cfg.output_dir, cfg.net_name, cfg.dataset, cfg.comp_algo))
    cfg.save_ckpt_dir = os.path.join(cfg.output_dir, 'ckpt')
    cfg.log_dir = os.path.join(cfg.output_dir, 'log')
    return cfg


def init_weight(net, cfg):
    """init_weight"""

    if cfg.pre_trained:
        if not os.path.isfile(cfg.pre_trained):
            cfg.logger.warning("There is not ckpt file: %s", cfg.pre_trained)
        else:
            param_dict = ms.load_checkpoint(cfg.pre_trained)
            if cfg.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(param_dict, filter_list)
            ms.load_param_into_net(net, param_dict)
            cfg.logger.info("Pre trained ckpt mode: %s loading", cfg.pre_trained)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if cfg.conv_init == "XavierUniform":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.XavierUniform(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif cfg.conv_init == "TruncatedNormal":
                    weight = conv_variance_scaling_initializer(cell.in_channels,
                                                               cell.out_channels,
                                                               cell.kernel_size[0])
                    cell.weight.set_data(weight)
            if isinstance(cell, nn.Dense):
                if cfg.dense_init == "TruncatedNormal":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif cfg.dense_init == "RandomNormal":
                    in_channel = cell.in_channels
                    out_channel = cell.out_channels
                    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                    weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                    cell.weight.set_data(weight)
