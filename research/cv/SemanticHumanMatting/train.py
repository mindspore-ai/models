# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train Semantic Human Matting Network and get checkpoint files."""
import os
import time
import random

import mindspore
from mindspore.context import ParallelMode
from mindspore.communication import init
from mindspore import Model, context, nn, DynamicLossScaleManager

from src.metric import Sad
from src.dataset import create_dataset
from src.model import network
from src.loss import LossTNet, LossMNet, LossNet
from src.load_model import load_pre_model
from src.callback import TrainCallBack, EvalCallBack, LossMonitorSub
from src.config import get_args, get_config_from_yaml, update_config


def init_env(cfg):
    """Init distribute env."""
    if cfg["saveIRFlag"]:
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target=cfg["device_target"],
            save_graphs=True,
            save_graphs_path=os.path.join(cfg["saveIRGraph"], cur_time),
        )
    else:
        context.set_context(
            mode=context.GRAPH_MODE, device_target=cfg["device_target"], reserve_class_name_in_scope=False
        )

    device_num = int(os.getenv("RANK_SIZE", "1"))
    cfg["group_size"] = device_num
    print(f"device_num:{device_num}")

    if cfg["device_target"] == "Ascend":
        devid = int(os.getenv("DEVICE_ID", "0"))
        cfg["rank"] = devid
        print(f"device_id:{devid}")
        context.set_context(device_id=devid)
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                parameter_broadcast=False,
            )
    else:
        raise ValueError("Unsupported platform.")


class CustomWithLossCell(nn.Cell):
    """
    Train network wrapper
    """

    def __init__(self, backbone, loss_fn_t, loss_fn_m, loss_fn, stage=0):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn_t = loss_fn_t
        self._loss_fn_m = loss_fn_m
        self._loss_fn = loss_fn
        self._stage = stage

    def construct(self, img, trimap_ch_gt, trimap_gt, alpha_gt):
        if self._stage == 0:
            trimap_pre = self._backbone(img)
            return self._loss_fn_t(trimap_pre, trimap_gt)
        if self._stage == 1:
            alpha_pre = self._backbone(img, trimap_ch_gt)
            return self._loss_fn_m(img, alpha_pre, alpha_gt)
        trimap_pre, alpha_pre = self._backbone(img)
        return self._loss_fn(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt)


class WithEvalCell(nn.Cell):
    """
    Evaluation network wrapper
    """

    def __init__(self, net, loss_fn_t_net, loss_fn_m_net, loss_fn, stage=0):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._backbone = net
        self._loss_fn_t_net = loss_fn_t_net
        self._loss_fn_m_net = loss_fn_m_net
        self._loss_fn = loss_fn
        self._stage = stage

    def construct(self, img, trimap_ch_gt, trimap_gt, alpha_gt):
        if self._stage == 0:
            trimap_pre = self._backbone(img)
            return trimap_pre, trimap_gt, self._loss_fn_t_net(trimap_pre, trimap_gt)
        if self._stage == 1:
            alpha_pre = self._backbone(img, trimap_ch_gt)
            return alpha_pre, alpha_gt, self._loss_fn_m_net(img, alpha_pre, alpha_gt)
        trimap_pre, alpha_pre = self._backbone(img)
        return alpha_pre, alpha_gt, self._loss_fn(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt)


def run_train(cfg):
    dataset_train, _ = create_dataset(cfg, "train", 1)
    dataset_eval, _ = create_dataset(cfg, "eval", 1)

    dict_stage = {"pre_train_t_net": 0, "pre_train_m_net": 1, "end_to_end": 2}
    net = network.net(stage=dict_stage[cfg["train_phase"]])
    cur_epoch = load_pre_model(net, cfg)
    print("----> total epoch: {}, current epoch: {}".format(str(cfg["nEpochs"]), str(cur_epoch)))
    if cfg["nEpochs"] <= cur_epoch:
        return

    net_loss = CustomWithLossCell(net, LossTNet(), LossMNet(), LossNet(), dict_stage[cfg["train_phase"]])
    net_eval = WithEvalCell(net, LossTNet(), LossMNet(), LossNet(), dict_stage[cfg["train_phase"]])

    scale_factor = 4
    scale_window = 3000
    loss_scale_manager = DynamicLossScaleManager(scale_factor, scale_window)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=float(cfg["lr"]), weight_decay=0.0005)
    model = Model(
        network=net_loss,
        optimizer=optim,
        metrics={"sad": Sad()},
        eval_network=net_eval,
        amp_level="O0",
        loss_scale_manager=loss_scale_manager,
    )

    if cfg["rank"] == 0:
        print("----> rank 0 is training.")
        call_back = TrainCallBack(
            cfg=cfg,
            network=net,
            model=model,
            eval_callback=[EvalCallBack()],
            eval_dataset=dataset_eval,
            cur_epoch=cur_epoch,
            per_print_times=1,
        )
    else:
        print("----> rank {} is training.".format(str(cfg["rank"])))
        call_back = LossMonitorSub(cur_epoch=cur_epoch)
    model.train(
        epoch=cfg["nEpochs"] - cur_epoch, train_dataset=dataset_train, callbacks=[call_back], dataset_sink_mode=False
    )


if __name__ == "__main__":
    args = get_args()
    config = get_config_from_yaml(args)
    random.seed(config["seed"])
    mindspore.set_seed(config["seed"])
    mindspore.common.set_seed(config["seed"])
    init_env(config)
    update_config(config)
    print("------------------------------config------------------------------")
    print(config)
    print("------------------------------------------------------------------")

    # Perform multistage train, the M-Net train phase is optional
    run_train(config["pre_train_t_net"])  # T-Net train phase
    # run_train(config['pre_train_m_net'])  # M-Net train phase
    run_train(config["end_to_end"])  # End-to-End train phase
