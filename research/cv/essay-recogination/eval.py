#!/bin/bash
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

import argparse
import gin
import editdistance

from src.utils import CTCLabelConverter
from src.cnv_model import OrigamiNet
from src import ds_load

import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import context
import mindspore.dataset as ds
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Model, Tensor
import mindspore.ops as ops


context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

def inference(model, img, converter):

    transpose = ops.Transpose()
    img = Tensor(img)

    preds = model.predict(img)

    preds_size = Tensor([preds.shape[1]], dtype=mstype.int32)
    preds = transpose(preds, (1, 0, 2))
    log_softmax = nn.LogSoftmax(axis=2)
    preds = log_softmax(preds)
    preds_index, _ = ops.ArgMaxWithValue(axis=2)(preds)
    preds_index = mnp.reshape(preds_index, -1)
    preds_index = preds_index.astype("int32").asnumpy()
    preds_str = converter.decodeOri(preds_index, preds_size)
    preds_str = converter.decode(preds_index, preds_size)
    return preds_str

@gin.configurable
def load_model(continue_model):
    model = OrigamiNet()
    param_dict = load_checkpoint(continue_model)
    load_param_into_net(model, param_dict)
    model = Model(model)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', default='./parameters/infer.gin', help='Gin config file')
    opt = parser.parse_args()

    gin.parse_config_file(opt.gin)
    eval_model = load_model()

    train_dataset = ds_load.myLoadDS(flist=None, dpath=None)
    eval_converter = CTCLabelConverter(train_dataset.ralph.values())
    val_data_d = ds_load.myLoadDS(flist="parameters/test.gc", dpath='data_set/eval/')

    val_data = ds.GeneratorDataset(source=val_data_d, column_names=["data", "label"])
    val_data.batch(batch_size=1)
    data_size = val_data.get_dataset_size()
    print("number of test data: ", data_size)

    val_data = val_data.create_tuple_iterator()
    norm_ED = 0
    for data in val_data:
        print("*******************************")
        string = inference(eval_model, data[0], eval_converter)
        s_l = list(string[0])
        preds_i = [val_data_d.alph[c] for c in s_l]
        print("preds result: ", preds_i)
        label = data[1]
        print("data label: ", label)
        tmped = editdistance.eval(preds_i, label)
        norm_ED += tmped / float(len(label))

    norm_ED /= data_size
    print("result: ", norm_ED)
