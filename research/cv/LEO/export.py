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
import mindspore as ms
from mindspore import context, Tensor, export
from mindspore import load_checkpoint, load_param_into_net
import numpy as np

os.environ['GLOG_v'] = "3"
os.environ['GLOG_log_dir'] = '/var/log'


def export_leo(init_config, inner_model_config, outer_model_config):
    inner_lr_init = inner_model_config['inner_lr_init']
    finetuning_lr_init = inner_model_config['finetuning_lr_init']

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

    batch = data_utils.get_batch('test')
    print(batch['train']['input'].shape)  # [200,5,5,640]
    print(batch['train']['input'].dtype)  # Float32
    print(batch['train']['target'].shape)  # [200,5,5,1]
    print(batch['train']['target'].dtype)  # Int64
    print(batch['val']['input'].shape)  # [200,5,15,640]
    print(batch['val']['input'].dtype)  # Float32
    print(batch['val']['target'].shape)  # [200,5,15,1]
    print(batch['val']['target'].dtype)  # Int64
    train_input = Tensor(np.zeros(batch['train']['input'].shape), ms.float32)
    train_target = Tensor(np.zeros(batch['train']['target'].shape), ms.int64)
    val_input = Tensor(np.zeros(batch['val']['input'].shape), ms.float32)
    val_target = Tensor(np.zeros(batch['val']['target'].shape), ms.int64)
    result_name = "LEO-" + init_config['dataset_name'] + str(outer_model_config['num_classes']) +\
                  "N" + str(outer_model_config['num_tr_examples_per_class']) + "K"
    export(test_outer_loop, train_input, train_target, val_input, val_target,
           file_name=result_name, file_format="MINDIR")


if __name__ == '__main__':
    initConfig = config.get_init_config()
    inner_model_Config = config.get_inner_model_config()
    outer_model_Config = config.get_outer_model_config()

    print("===============inner_model_config=================")
    for key, value in inner_model_Config.items():
        print(key + ": " + str(value))
    print("===============outer_model_config=================")
    for key, value in outer_model_Config.items():
        print(key + ": " + str(value))

    context.set_context(mode=context.GRAPH_MODE, device_target=initConfig['device_target'])

    export_leo(initConfig, inner_model_Config, outer_model_Config)
    print("successfully export LEO model!")
