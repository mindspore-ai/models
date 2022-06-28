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
import argparse
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore import context, Model, DynamicLossScaleManager
from mindspore.communication.management import init, get_group_size
from mindspore.nn.optim import AdamWeightDecay
from src.finetune_config import optimizer_cfg, bert_cfg
from src.tokenization import CscTokenizer
from src.soft_masked_bert import SoftMaskedBertCLS

def do_train(dataset, network, cfg, profile=None, save_ckpt_path='./checkpoint', epoch_num=1):
    """ do train """
    max_epoch = 100
    steps_per_epoch = dataset.get_dataset_size()
    # network.to_float(mstype.float16)
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        params = network.trainable_params()
        bias_params = list(filter(lambda x: 'bias' in x.name, params))
        no_bias_params = list(filter(lambda x: 'bias' not in x.name, params))
        group_params = [{'params': bias_params, 'weight_decay': 0, 'lr': cfg.baselr * cfg.bias_lr_factor},
                        {'params': no_bias_params, 'weight_decay': cfg.weight_decay, 'lr': cfg.baselr}]
        optimizer = AdamWeightDecay(group_params, learning_rate=cfg.baselr)
    if cfg.enable_modelarts:
        config = CheckpointConfig(saved_network=network, save_checkpoint_steps=steps_per_epoch * max_epoch)
    else:
        config = CheckpointConfig(saved_network=network, save_checkpoint_steps=steps_per_epoch * 10)
    ckpoint_cb = ModelCheckpoint(prefix='SoftMaskedBert',
                                 directory=save_ckpt_path,
                                 config=config)
    time_cb = TimeMonitor(data_size=steps_per_epoch)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2**24)
    model = Model(network, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")
    model.train(epoch=max_epoch, train_dataset=dataset, callbacks=[LossMonitor(), \
    ckpoint_cb, time_cb], dataset_sink_mode=False)
    if cfg.enable_modelarts:
        import moxing as mox
        mox.file.copy_parallel(save_ckpt_path, cfg.train_url)

def run_csc():
    """run csc task"""
    parser = argparse.ArgumentParser(description="run csc")
    parser.add_argument("--bert_ckpt", type=str, \
    default="bert_base.ckpt")
    parser.add_argument("--device_target", type=str, default="Ascend")
    parser.add_argument("--name", type=str, default="SoftMaskedBertModel")
    parser.add_argument("--hyper_params", type=float, default=0.8)
    parser.add_argument("--baselr", type=float, default=0.00001) # 0.0001
    parser.add_argument("--bias_lr_factor", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=5e-8)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=bert_cfg.seq_length) #512
    parser.add_argument("--train_url", type=str, default="./datasets/csc")  # output direction, such as s3://open-data/job/openizxche2022062222t062300200037543/output/V0012/
    parser.add_argument("--data_url", type=str, default="./datasets/csc/train.json")  # direction of the training dataset, such as s3://open-data/attachment/1/4/1493e5f0-4601-408e-bc7f-b51ef8b3785c1493e5f0-4601-408e-bc7f-b51ef8b3785c/
    def str2bool(input_str):
        return bool(input_str)
    parser.add_argument("--enable_modelarts", type=str2bool, default='False')
    parser.add_argument("--pynative", type=str2bool, default='False')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--rank_size", type=int, default=1)
    args_opt = parser.parse_args()
    if args_opt.enable_modelarts:
        import moxing as mox
        if mox.file.exists('/cache/dataset'):
            ret = mox.file.list_directory('/cache/dataset', recursive=True)
            print('/cache/dataseet: (recursive)')
            print(ret)
        cloud_data_url = args_opt.data_url
        local_root_dir = '/home/work/user-job-dir'
        local_data_dir = os.path.join(local_root_dir, "data")
        local_train_file_dir = os.path.join(local_data_dir, "SoftMask", "train.json")
        local_ckpt_dir = os.path.join(local_data_dir, "SoftMask", args_opt.bert_ckpt)
        local_vocab_dir = os.path.join(local_data_dir, "SoftMask", "bert-base-chinese-vocab.txt")
        local_model_dir = os.path.join(local_root_dir, "model")
        if mox.file.exists(local_data_dir) is False:
            mox.file.make_dirs(local_data_dir)
        if mox.file.exists(local_model_dir) is False:
            mox.file.make_dirs(local_model_dir)
        mox.file.copy_parallel(cloud_data_url, local_data_dir)
        print(local_data_dir + ":")
        ret = mox.file.list_directory(local_data_dir, recursive=True)
        print(ret)
    else:
        local_ckpt_dir = './weight/' + args_opt.bert_ckpt
        local_model_dir = './checkpoint'
        local_data_dir = args_opt.data_url
        local_train_file_dir = local_data_dir
    # context setting
    if args_opt.device_target != "Ascend":
        raise Exception("Only support on Ascend currently.")
    run_mode = context.GRAPH_MODE
    if args_opt.pynative:
        run_mode = context.PYNATIVE_MODE
        args_opt.batch_size = 16
    device_id = args_opt.device_id
    device_num = args_opt.rank_size
    if args_opt.enable_modelarts:
        device_id = int(os.environ["DEVICE_ID"])
        init()
        device_num = get_group_size()
    context.set_context(mode=run_mode, device_target="Ascend", device_id=device_id)
    if args_opt.enable_modelarts:
        context.set_auto_parallel_context(device_num=device_num, \
        gradients_mean=True, parallel_mode=context.ParallelMode.DATA_PARALLEL)
    netwithloss = SoftMaskedBertCLS(args_opt.batch_size, is_training=True, load_checkpoint_path=local_ckpt_dir)
    netwithloss.set_train(True)
    if args_opt.enable_modelarts:
        tokenizer = CscTokenizer(fp=local_train_file_dir, device_num=device_num, rank_id=device_id, \
                                max_seq_len=args_opt.max_seq_len, vocab_path=local_vocab_dir)
    else:
        tokenizer = CscTokenizer(fp=local_train_file_dir, device_num=device_num, rank_id=device_id, \
                                max_seq_len=args_opt.max_seq_len)
    ds_train = tokenizer.get_token_ids(args_opt.batch_size)
    do_train(ds_train, netwithloss, cfg=args_opt, save_ckpt_path=local_model_dir)
if __name__ == "__main__":
    run_csc()
