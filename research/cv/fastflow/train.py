# Copyright 2022 Huawei Technologies Co., Ltd
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
"""train"""
import os
import ast
import time
from sklearn.metrics import roc_auc_score

import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint

from src.resnet import wide_resnet50_2
from src.dataset import createDataset
from src.fastflow import build_model
from src.loss import NetWithLossCell, FastflowLoss
from src.utils import AverageMeter
from src.config import get_arguments
from src.cell import TrainOneStepCell
from src.anomaly_map import anomaly_map_generator

set_seed(1)

def preLauch():
    """parse the console argument"""
    parser = get_arguments()
    # Train Device.
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
    parser.add_argument('--device_target', type=str, default='Ascend')
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')

    return parser.parse_args()

def train_one_epoch(cfg, train_dataset_epoch, model_ms, cur_epoch):
    model_ms.set_train(True)
    model_ms.network.network.feature_extractor.set_train(False)
    data_iter = train_dataset_epoch.create_dict_iterator()
    step_size = train_dataset_epoch.get_dataset_size()
    loss_meter = AverageMeter()
    for step, data in enumerate(data_iter):
        # time
        start = time.time()
        cur_loss = model_ms(data['img'])
        end = time.time()
        step_time = (end - start) * 1000

        # log
        loss_meter.update(cur_loss.asnumpy())
        if (step + 1) % cfg.log_interval == 0 or (step + 1) == step_size:
            print(
                "Epoch {} - Step {}: time: {}ms, loss = {:.3f}({:.3f})".format(
                    cur_epoch + 1, step + 1, step_time, loss_meter.val, loss_meter.avg
                )
            )

def eval_once(test_dataset_epoch, network_ms, cur_epoch, output_size=256):
    network_ms.set_train(False)
    data_iter = test_dataset_epoch.create_dict_iterator()

    gt_list_px_lvl = []
    pred_list_px_lvl = []
    for _, data in enumerate(data_iter):
        hidden_variables, _ = network_ms(data['img'])
        anomaly_map = anomaly_map_generator(hidden_variables, output_size=output_size)

        gt_np = data['gt'].asnumpy().astype(int)
        gt = gt_np[:, 0]
        gt_list_px_lvl.extend(gt.ravel())
        pred_list_px_lvl.extend(anomaly_map.ravel())

    cur_pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    print("Epoch {}: pixel_auc: {:.4f}".format(cur_epoch + 1, cur_pixel_auc))

    return cur_pixel_auc

if __name__ == '__main__':
    args = preLauch()
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE, save_graphs=False)
    context.set_context(device_id=args.device_id)

    # dataset
    train_dataset, test_dataset = createDataset(
        dataset_path=args.dataset_path,
        category=args.category,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        im_resize=args.im_resize
        )

    # network
    feature_extractor = wide_resnet50_2()
    param_dict = load_checkpoint(args.pre_ckpt_path)
    load_param_into_net(feature_extractor, param_dict)

    for p in feature_extractor.trainable_params():
        p.requires_grad = False

    network = build_model(
        backbone=feature_extractor,
        flow_step=args.flow_step,
        im_resize=args.im_resize,
        conv3x3_only=args.conv3x3_only,
        hidden_ratio=args.hidden_ratio
        )
    loss = FastflowLoss()
    model = NetWithLossCell(network, loss)

    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    TrainOneStep = TrainOneStepCell(model, optimizer)

    # ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "fastflow_%s.ckpt" % (args.category))

    # train
    best_pixel_auc = 0.0
    print("***************start train***************")
    for epoch in range(args.num_epochs):
        train_one_epoch(args, train_dataset, TrainOneStep, epoch)
        if (epoch + 1) % args.eval_interval == 0:
            pixel_auc = eval_once(test_dataset, network, epoch, output_size=args.im_resize)

            if pixel_auc > best_pixel_auc:
                best_pixel_auc = pixel_auc
                save_checkpoint(network, ckpt_path)
    print("***************train end***************")
    print("for category {} : best_pixel_auc = {:.4f}".format(args.category, best_pixel_auc))
