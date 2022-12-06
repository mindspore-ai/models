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
'''eval dataset with SGAE model'''
import os
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context, Tensor
from mindspore import dtype as mstype
from src.SGAE import SGAE
from dataloader import LoadDocumentData, LoadImageData, LoadTabularData
from model_utils.config import config as cfg

data_name = ""

def eval_tabular(params):
    ''' eval tabular '''
    #Load data
    x_train, _, _, _, x_test, y_test = LoadTabularData(params)
    x_train_whole = Tensor(x_train, dtype=mstype.float32)
    x_test_whole = Tensor(x_test, dtype=mstype.float32)
    global data_name
    data_name = params.data_name
    run_index = params.run_idx

    model = SGAE(x_train_whole.shape[1], params.hidden_dim)
    model.set_train(False)
    param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt")
    load_param_into_net(model, param_dict)
    scores, _, _ = model(x_test_whole)
    scores = scores.asnumpy()
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)

    print(f'val finished, AUC={auc:.3f}, AP={ap:.3f}')
    return {'AUC': f'{auc:.3f}', 'AP': f'{ap:.3f}'}

def eval_image(params):
    ''' eval image '''
    #Load data
    x_train, x_test, _, y_test = LoadImageData(params)
    x_train_whole = Tensor(x_train, dtype=mstype.float32)
    x_test_whole = Tensor(x_test, dtype=mstype.float32)
    global data_name
    data_name = params.data_name
    run_index = params.run_idx
    model = SGAE(x_train_whole.shape[1], params.hidden_dim)
    model.set_train(False)
    param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt")
    load_param_into_net(model, param_dict)
    scores, _, _ = model(x_test_whole)
    scores = scores.asnumpy()
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    print(f'eval finished, AUC={auc:.3f}, AP={ap:.3f}')
    return {'AUC': f'{auc:.3f}', 'AP': f'{ap:.3f}'}

def eval_document(params):
    ''' eval document '''
    dataloader = LoadDocumentData(params)
    auc = np.zeros((dataloader.class_num,))
    ap = np.zeros((dataloader.class_num,))
    global data_name
    data_name = params.data_name
    run_idx = params.run_idx

    for normal_idx in range(dataloader.class_num):
        x_train, x_test, _, y_test = dataloader.preprocess(normal_idx)
        x_train_whole = Tensor(x_train, dtype=mstype.float32)
        x_test_whole = Tensor(x_test, dtype=mstype.float32)
        model = SGAE(x_train_whole.shape[1], params.hidden_dim)

        if params.verbose and normal_idx == 0 and run_idx == 0:
            print(model)

        param_dict = \
        load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_idx}_normal{normal_idx}.ckpt")
        load_param_into_net(model, param_dict)
        scores, _, _ = model(x_test_whole)
        scores = scores.asnumpy()
        auc[normal_idx] = roc_auc_score(y_test, scores)
        ap[normal_idx] = average_precision_score(y_test, scores)
        print(f'this idx finished, AUC={np.mean(auc[normal_idx]):.3f}, AP={np.mean(ap[normal_idx]):.3f}')
    return {'AUC': f'{np.mean(auc):.3f}', 'AP': f'{np.mean(ap):.3f}'}

if __name__ == '__main__':
    start_time = time.time()
    time_name = str(time.strftime("%m%d")) + '_' + str(time.time()).split(".")[1][-3:]
    print(f'Time name is {time_name}')
    print(os.getcwd())
    metrics = pd.DataFrame()
    args = cfg

    if args.ms_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE)
    else:
        context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(device_target=args.device, device_id=0)
    if args.data_name in ['attack', 'bcsc', 'creditcard', 'diabetic', 'donor', 'intrusion', 'market']:
        an_metrics_dict = eval_tabular(args)
    elif args.data_name in ['reuters', '20news']:
        an_metrics_dict = eval_document(args)
    elif args.data_name in ['mnist']:
        an_metrics_dict = eval_image(args)

    metrics = pd.DataFrame(an_metrics_dict, index=[0])
    metrics.to_csv(f'{args.out_dir}{args.model_name}_{args.data_name}_{time_name}.csv')
    print(f'Finished!\nTotal time is {time.time()-start_time:.2f}s')
    print(f'Current time is {time.strftime("%m%d_%H%M")}')
    print(f'Results:')
    print(metrics.sort_values('AUC', ascending=False))
    