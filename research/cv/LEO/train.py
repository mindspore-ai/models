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

import os
import time
import model_utils.config as config
import src.data as data
import src.outerloop as outerloop
from src.trainonestepcell import TrainOneStepCell
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import save_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode


os.environ['GLOG_v'] = "3"
os.environ['GLOG_log_dir'] = '/var/log'


def save_checkpoint_to_file(if_save_checkpoint, val_accs, best_acc, step, val_losses, init_config, train_outer_loop):
    if if_save_checkpoint:
        if not sum(val_accs) / len(val_accs) < best_acc:
            best_acc = sum(val_accs) / len(val_accs)
            model_name = '%dk_%4.4f_%4.4f_model.ckpt' % (
                (step // 1000 + 1),
                sum(val_losses) / len(val_losses),
                sum(val_accs) / len(val_accs))

            check_dir(init_config['save_path'])

            if args.enable_modelarts:
                save_checkpoint_path = '/cache/train_output/device_' + \
                                       os.getenv('DEVICE_ID') + '/'
                save_checkpoint_path = '/cache/train_output/'
                if not os.path.exists(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)
                save_checkpoint(train_outer_loop, os.path.join(save_checkpoint_path, model_name))
            else:
                save_checkpoint(train_outer_loop, os.path.join(init_config['save_path'], model_name))
            print('Saved checkpoint %s...' % model_name)


def train_leo(init_config, inner_model_config, outer_model_config):
    inner_lr_init = inner_model_config['inner_lr_init']
    finetuning_lr_init = inner_model_config['finetuning_lr_init']

    total_train_steps = outer_model_config['total_steps']
    val_every_step = 3000
    total_val_steps = 100
    if_save_checkpoint = True
    best_acc = 0
    sum_steptime = 0

    data_utils = data.Data_Utils(
        train=True, seed=100, way=outer_model_config['num_classes'],
        shot=outer_model_config['num_tr_examples_per_class'],
        data_path=init_config['data_path'], dataset_name=init_config['dataset_name'],
        embedding_crop=init_config['embedding_crop'],
        batchsize=outer_model_config['metatrain_batch_size'],
        val_batch_size=outer_model_config['metavalid_batch_size'],
        test_batch_size=outer_model_config['metatest_batch_size'],
        meta_val_steps=outer_model_config['num_val_examples_per_class'], embedding_size=640, verbose=True)

    train_outer_loop = outerloop.OuterLoop(
        batchsize=outer_model_config['metatrain_batch_size'],
        input_size=640, latent_size=inner_model_config['num_latents'],
        way=outer_model_config['num_classes'], shot=outer_model_config['num_tr_examples_per_class'],
        dropout=inner_model_config['dropout_rate'], kl_weight=inner_model_config['kl_weight'],
        encoder_penalty_weight=inner_model_config['encoder_penalty_weight'],
        orthogonality_weight=inner_model_config['orthogonality_penalty_weight'],
        inner_lr_init=inner_lr_init, finetuning_lr_init=finetuning_lr_init,
        inner_step=inner_model_config['inner_unroll_length'],
        finetune_inner_step=inner_model_config['finetuning_unroll_length'], is_meta_training=True)

    val_outer_loop = outerloop.OuterLoop(
        batchsize=outer_model_config['metavalid_batch_size'],
        input_size=640, latent_size=inner_model_config['num_latents'],
        way=outer_model_config['num_classes'], shot=outer_model_config['num_tr_examples_per_class'],
        dropout=inner_model_config['dropout_rate'], kl_weight=inner_model_config['kl_weight'],
        encoder_penalty_weight=inner_model_config['encoder_penalty_weight'],
        orthogonality_weight=inner_model_config['orthogonality_penalty_weight'],
        inner_lr_init=inner_lr_init, finetuning_lr_init=finetuning_lr_init,
        inner_step=inner_model_config['inner_unroll_length'],
        finetune_inner_step=inner_model_config['finetuning_unroll_length'], is_meta_training=True)

    if context.get_context("device_target") == "Ascend":
        train_outer_loop.to_float(mindspore.float32)
        for _, cell in train_outer_loop.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mindspore.float16)

    train_net = TrainOneStepCell(train_outer_loop,
                                 outer_model_config['outer_lr'],
                                 inner_model_config['l2_penalty_weight'])


    for step in range(total_train_steps):
        if step == 0:
            train_start = time.time()
        old_t = time.time()
        if step == 50000:
            train_net = TrainOneStepCell(train_outer_loop,
                                         outer_model_config['outer_lr']/2,
                                         inner_model_config['l2_penalty_weight'])
        batch = data_utils.get_batch('train')
        val_loss, val_acc = train_net(batch['train']['input'],
                                      batch['train']['target'],
                                      batch['val']['input'],
                                      batch['val']['target'],
                                      train=True)
        now_t = time.time()
        sum_steptime += (now_t-old_t)
        print('(Meta-Train)[Step: %d/%d] Total Loss: %4.4f \
               Inner_Lr: %4.4f Finetuning_Lr: %4.4f \
               Valid Accuracy: %4.4f step_time: %4.4f'%(step,
                                                        total_train_steps,
                                                        val_loss.asnumpy(),
                                                        train_net.group_params[0]['params'][0].T.asnumpy(),
                                                        train_net.group_params[0]['params'][1].T.asnumpy(),
                                                        val_acc.asnumpy(), now_t-old_t))

        if step % val_every_step == 2999:
            print('3000 step average time: %4.4f second...'%(sum_steptime/3000))
            sum_steptime = 0

            val_losses = []
            val_accs = []
            train_parms = train_outer_loop.parameters_dict()
            load_param_into_net(val_outer_loop, train_parms)
            for _ in range(total_val_steps):
                batch = data_utils.get_batch('val')
                val_loss, val_acc = val_outer_loop(batch['train']['input'],
                                                   batch['train']['target'],
                                                   batch['val']['input'],
                                                   batch['val']['target'],
                                                   train=False)
                val_losses.append(val_loss.asnumpy())
                val_accs.append(val_acc.asnumpy())

            print('=' * 50)
            print('Meta Valid Loss: %4.4f Meta Valid Accuracy: %4.4f'%
                  (sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs)))
            print('=' * 50)

            save_checkpoint_to_file(if_save_checkpoint, val_accs, best_acc, step, val_losses,
                                    init_config, train_outer_loop)

        if step == (total_train_steps-1):
            train_end = time.time()
            print('Spend total time: %d minute...'%((train_end-train_start)/60))


def check_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Create dir %s/'%save_path)
    else:
        ckpt_list = os.listdir(save_path)
        if len(ckpt_list) <= 9:
            pass
        else:
            ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime \
                                (os.path.join(save_path, x)))
            for i in range(len(ckpt_list) - 9):
                print('Remove checkpoint %s...'%ckpt_list[i])
                os.remove(os.path.join(save_path, ckpt_list[i]))


if __name__ == '__main__':
    initConfig = config.get_init_config()
    inner_model_Config = config.get_inner_model_config()
    outer_model_Config = config.get_outer_model_config()
    args = config.get_config(get_args=True)


    print("===============inner_model_config=================")
    for key, value in inner_model_Config.items():
        print(key+": "+str(value))
    print("===============outer_model_config=================")
    for key, value in outer_model_Config.items():
        print(key+": "+str(value))

    context.set_context(mode=context.GRAPH_MODE, device_target=initConfig['device_target'])
    if args.enable_modelarts:
        import moxing as mox

        mox.file.copy_parallel(
            src_url=args.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = os.path.join('/cache/dataset/device_' + os.getenv('DEVICE_ID'), "embeddings")
        initConfig['data_path'] = train_dataset_path

    elif initConfig['device_num'] > 1:
        init('nccl')
        context.set_auto_parallel_context(device_num=initConfig['device_num'],
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    train_leo(initConfig, inner_model_Config, outer_model_Config)
    if args.enable_modelarts:
        mox.file.copy_parallel(
            src_url='/cache/train_output', dst_url=args.train_url)
