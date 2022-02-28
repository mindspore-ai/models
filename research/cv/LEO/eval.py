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
import src.data as data
import src.outerloop as outerloop
import model_utils.config as config
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from tqdm.std import tqdm
import numpy as np


os.environ['GLOG_v'] = "3"
os.environ['GLOG_log_dir'] = '/var/log'


def eval_leo(init_config, inner_model_config, outer_model_config):

    inner_lr_init = inner_model_config['inner_lr_init']
    finetuning_lr_init = inner_model_config['finetuning_lr_init']

    total_test_steps = 100

    data_utils = data.Data_Utils(
        train=False, seed=100, way=outer_model_config['num_classes'],
        shot=outer_model_config['num_tr_examples_per_class'],
        data_path=init_config['data_path'], dataset_name=init_config['dataset_name'],
        embedding_crop=init_config['embedding_crop'],
        batchsize=outer_model_config['metatrain_batch_size'],
        val_batch_size=outer_model_config['metavalid_batch_size'],
        test_batch_size=outer_model_config['metatest_batch_size'],
        meta_val_steps=outer_model_config['num_val_examples_per_class'], embedding_size=640, verbose=True)

    test_outer_loop = outerloop.OuterLoop(
        batchsize=outer_model_config['metavalid_batch_size'], input_size=640,
        latent_size=inner_model_config['num_latents'],
        way=outer_model_config['num_classes'], shot=outer_model_config['num_tr_examples_per_class'],
        dropout=inner_model_config['dropout_rate'], kl_weight=inner_model_config['kl_weight'],
        encoder_penalty_weight=inner_model_config['encoder_penalty_weight'],
        orthogonality_weight=inner_model_config['orthogonality_penalty_weight'],
        inner_lr_init=inner_lr_init, finetuning_lr_init=finetuning_lr_init,
        inner_step=inner_model_config['inner_unroll_length'],
        finetune_inner_step=inner_model_config['finetuning_unroll_length'], is_meta_training=False)

    parm_dict = load_checkpoint(init_config['ckpt_file'])
    load_param_into_net(test_outer_loop, parm_dict)

    test_losses = []
    test_accs = []

    for _ in tqdm(range(total_test_steps)):
        batch = data_utils.get_batch('test')
        test_loss, test_acc = test_outer_loop(batch['train']['input'], batch['train']['target'],
                                              batch['val']['input'], batch['val']['target'], train=False)
        test_losses.append(test_loss.asnumpy())
        test_accs.append(test_acc.asnumpy())
    interval = 1.96*np.sqrt(np.var(test_accs)/len(test_accs))
    print('Meta Valid Accuracy: %4.4f±%4.4f'%(sum(test_accs)/len(test_accs), interval))


if __name__ == '__main__':
    initConfig = config.get_init_config()
    inner_model_Config = config.get_inner_model_config()
    outer_model_Config = config.get_outer_model_config()

    print("===============inner_model_config=================")
    for key, value in inner_model_Config.items():
        print(key+": "+str(value))
    print("===============outer_model_config=================")
    for key, value in outer_model_Config.items():
        print(key+": "+str(value))

    context.set_context(mode=context.GRAPH_MODE, device_target=initConfig['device_target'])

    eval_leo(initConfig, inner_model_Config, outer_model_Config)
