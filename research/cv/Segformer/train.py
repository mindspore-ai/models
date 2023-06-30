# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import time
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from eval import do_eval
from src.loss import CrossEntropy
from src.model_utils.config import get_train_config
from src.segformer import SegFormer
from src.dataset import prepare_cityscape_dataset, get_train_dataset, get_eval_dataset
from src.optimizers import get_optimizer
from src.model_utils.common import current_time, rank_sync


@rank_sync
def epoch_end_process(config, epoch, net, eval_model, eval_dataset_iterator, eval_dataset_size, result):
    os.makedirs(config.checkpoint_path, exist_ok=True)
    ckpt_name = "segformer_{}_{}.ckpt".format(config.backbone, epoch)
    ckpt_path = os.path.join(config.checkpoint_path, ckpt_name)
    best_ckpt_name = "segformer_{}_best.ckpt".format(config.backbone)
    best_ckpt_path = os.path.join(config.checkpoint_path, best_ckpt_name)
    ckpt_saved_flag = False
    ckpt_saved_best_flag = False

    if config.run_eval and epoch >= config.eval_start_epoch \
            and epoch % config.eval_interval == 0 and config.rank_id == 0:
        ms.save_checkpoint(net, ckpt_path)
        config.eval_ckpt_path = ckpt_path
        ckpt_saved_flag = True
        _param_dict = {}
        for _p in net.get_parameters():
            _param_dict[_p.name] = _p.data
        ms.load_param_into_net(eval_model, _param_dict)
        del _param_dict
        mean_iou = do_eval(config, net=eval_model, dataset_iterator=eval_dataset_iterator,
                           dataset_size=eval_dataset_size)
        if mean_iou > result["best_mean_iou"]:
            result["best_mean_iou"] = mean_iou
            result["ckpt_name"] = ckpt_name
            if config.save_best_ckpt:
                ms.save_checkpoint(net, best_ckpt_path)
                ckpt_saved_best_flag = True

    if config.save_checkpoint and epoch % config.save_checkpoint_epoch_interval == 0 \
            and config.rank_id == 0 and not ckpt_saved_flag:
        ms.save_checkpoint(net, ckpt_path)
        ckpt_saved_flag = True

    if config.enable_modelarts:
        if ckpt_saved_flag:
            sync_data(ckpt_path, os.path.join(config.train_url, ckpt_name))
        if ckpt_saved_best_flag:
            sync_data(best_ckpt_path, os.path.join(config.train_url, best_ckpt_name))


def do_train(config):
    train_begin_time = int(time.time())
    class_num = config.class_num

    prepare_cityscape_dataset(config.data_path)
    train_dataset = get_train_dataset(config)
    dataset_size = train_dataset.get_dataset_size()
    print(f"{current_time()}, Rank {config.rank_id}, train data size:{dataset_size}")
    train_net = SegFormer(config.backbone, class_num, sync_bn=config.run_distribute).to_float(ms.float16)

    eval_net = None
    eval_dataset_size = None
    eval_dataset_iterator = None
    if config.run_eval and rank_id == 0:
        eval_dataset = get_eval_dataset(config)
        eval_dataset_size = eval_dataset.get_dataset_size()
        eval_dataset_iterator = eval_dataset.create_dict_iterator()
        eval_net = SegFormer(config.backbone, class_num, sync_bn=config.run_distribute).to_float(ms.float16)
        eval_net.set_train(False)

    if config.load_ckpt:
        if os.path.exists(config.pretrained_ckpt_path):
            param_dict = ms.load_checkpoint(config.pretrained_ckpt_path)
            ms.load_param_into_net(train_net, param_dict)
            print(f"{current_time()}, Rank {config.rank_id}, load {config.pretrained_ckpt_path} success.")
        else:
            raise Exception(f"the pretrained model: {config.pretrained_ckpt_path} is not exists.")

    loss_fn = CrossEntropy(num_classes=class_num, ignore_label=255)

    optimizer = get_optimizer(train_net, config.optimizer, lr=config.lr, epochs=config.epoch_size,
                              weight_decay=config.weight_decay, step_per_epoch=dataset_size,
                              warmup_steps=config.warmup_steps)

    net_with_loss = nn.WithLossCell(train_net, loss_fn)
    loss_scaler = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 14, scale_factor=2, scale_window=1000)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, loss_scaler)

    result = {"best_mean_iou": 0, "ckpt_name": ""}
    iterator = train_dataset.create_dict_iterator()
    for epoch in range(0, config.epoch_size):
        step_begin_time = int(time.time() * 1000)
        for step_idx, item in enumerate(iterator):
            image = item['image']
            label = item['label']
            loss, overflow, loss_scale = train_net(image, label)
            step_end_time = int(time.time() * 1000)
            if (step_idx + 1) % config.train_log_interval == 0:
                print(f"{current_time()}, Rank {config.rank_id}, Epoch {epoch + 1}/{config.epoch_size}, "
                      f"step:{step_idx + 1}/{dataset_size}, "
                      f"loss:{loss.asnumpy():.6f}, overflow:{overflow}, loss_scale:{loss_scale.asnumpy()}, "
                      f"step cost:{step_end_time - step_begin_time}ms")
            step_begin_time = step_end_time
        train_dataset.reset()
        epoch_end_process(config, epoch + 1, train_net, eval_net, eval_dataset_iterator, eval_dataset_size, result)

    print(f"{current_time()}, Rank {config.rank_id}, best result: {result}")
    train_end_time = int(time.time())
    print(f"{current_time()}, Rank {config.rank_id}, all train process done, cost:{train_end_time - train_begin_time}s")


if __name__ == '__main__':
    train_config = get_train_config()
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = int(os.getenv('DEVICE_NUM', '1'))
    print(f"device_id:{device_id}, device_num:{device_num}")

    if device_id % 8 == 0:
        os.system("rm -f /tmp/segformer_sync.lock*")

    context.set_context(mode=context.GRAPH_MODE, device_target=train_config.device_target, device_id=device_id)
    rank_id, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if train_config.run_distribute:
        init()
        rank_id, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)
    train_config.rank_id = rank_id
    train_config.rank_size = rank_size
    train_config.total_batch_size = train_config.batch_size
    if rank_size > 1:
        assert train_config.batch_size % rank_size == 0, '--batch_size must be multiple of device count'
        train_config.batch_size = train_config.total_batch_size // rank_size
    print(f"train config:{train_config}")

    if train_config.enable_modelarts:
        from src.model_utils.modelarts import sync_data
        print(f"data_url:{train_config.data_url}, train_url:{train_config.train_url}, "
              f"data_dir:{train_config.data_dir}")
        os.makedirs(train_config.data_dir, exist_ok=True)
        sync_data(train_config.data_url, train_config.data_dir)

    do_train(train_config)
