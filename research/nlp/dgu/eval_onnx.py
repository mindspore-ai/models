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

'''
Bert finetune and evaluation script.
'''

import os
import time
import mindspore.common.dtype as mstype
import mindspore.ops as P

from mindspore import context, Tensor
from mindspore import log as logger
from mindspore.nn import Accuracy

import src.dataset as data
import src.metric as metric
from src.args import parse_args, set_default_args
from src.finetune_eval_config import (bert_net_cfg, bert_net_udc_cfg)
from src.utils import (create_classification_dataset, make_directory)

import onnxruntime as ort


def eval_result_print(eval_metric, result):
    if args_opt.task_name.lower() in ['atis_intent', 'mrda', 'swda']:
        metric_name = "Accuracy"
    else:
        metric_name = eval_metric.name()
    print(metric_name, " :", result)
    if args_opt.task_name.lower() == "udc":
        print("R1@10: ", result[0])
        print("R2@10: ", result[1])
        print("R5@10: ", result[2])


def create_session(checkpoint_path, target_device):
    # create onnx session
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU"')
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def do_eval(dataset=None, eval_metric=None, load_onnx_path=""):
    """ do eval """

    print("eval model: ", load_onnx_path)
    print("loading... ")
    session, [input_ids_name, input_mask_name, token_type_id_name] = create_session(load_onnx_path, 'GPU')

    print("evaling... ")
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    eval_metric.clear()
    evaluate_times = []
    for data_item in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data_item[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        squeeze = P.Squeeze(-1)
        label_ids = squeeze(label_ids)
        time_begin = time.time()
        logits = session.run(None, {input_ids_name: input_ids.asnumpy(), input_mask_name: input_mask.asnumpy(),
                                    token_type_id_name: token_type_id.asnumpy()})[0]
        time_end = time.time()
        evaluate_times.append(time_end - time_begin)
        logits = Tensor(logits)

        eval_metric.update(logits, label_ids)


    print("==============================================================")
    print("(w/o first and last) elapsed time: {}, per step time : {}".format(
        sum(evaluate_times[1:-1]), sum(evaluate_times[1:-1])/(len(evaluate_times) - 2)))
    print("==============================================================")
    result = eval_metric.eval()
    eval_result_print(eval_metric, result)
    return result


def eval_onnx(args_input):
    """run_dgu main function """
    metric_class = TASK_CLASSES[args_input.task_name][1]

    target = args_input.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_input.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_input.device_id)
        if net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")



    eval_ds = create_classification_dataset(batch_size=args_input.eval_batch_size, repeat_count=1, \
                data_file_path=args_input.eval_data_file_path, \
                do_shuffle=(args_input.eval_data_shuffle.lower() == "true"), drop_remainder=True)
    if args_input.task_name in ['atis_intent', 'mrda', 'swda']:
        eval_metric = metric_class("classification")
    else:
        eval_metric = metric_class()
    #load model from path and eval
    if args_input.eval_ckpt_path:
        do_eval(eval_ds, eval_metric, args_input.eval_ckpt_path)

    else:
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")


def print_args_input(args_input):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args_input).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def set_bert_cfg():
    """set bert cfg"""
    global net_cfg
    global eval_net_cfg
    if args_opt.task_name == 'udc':
        net_cfg = bert_net_udc_cfg
        eval_net_cfg = bert_net_udc_cfg
        print("use udc_bert_cfg")
    else:
        net_cfg = bert_net_cfg
        eval_net_cfg = bert_net_cfg
    return net_cfg, eval_net_cfg

if __name__ == '__main__':
    TASK_CLASSES = {
        'udc': (data.UDCv1, metric.RecallAtK),
        'atis_intent': (data.ATIS_DID, Accuracy),
        'mrda': (data.MRDA, Accuracy),
        'swda': (data.SwDA, Accuracy)
    }
    os.environ['GLOG_v'] = '3'
    eval_file_dict = {}
    args_opt = parse_args()
    set_default_args(args_opt)
    net_cfg, eval_net_cfg = set_bert_cfg()
    load_pretrain_checkpoint_path = args_opt.model_name_or_path
    save_finetune_checkpoint_path = args_opt.checkpoints_path + args_opt.task_name
    save_finetune_checkpoint_path = make_directory(save_finetune_checkpoint_path)
    if args_opt.is_modelarts_work == 'true':
        import moxing as mox
        local_load_pretrain_checkpoint_path = args_opt.local_model_name_or_path
        local_data_path = '/cache/data/' + args_opt.task_name
        mox.file.copy_parallel(args_opt.data_url + args_opt.task_name, local_data_path)
        mox.file.copy_parallel('obs:/' + load_pretrain_checkpoint_path, local_load_pretrain_checkpoint_path)
        load_pretrain_checkpoint_path = local_load_pretrain_checkpoint_path
        if not args_opt.train_data_file_path:
            args_opt.train_data_file_path = local_data_path + '/' + args_opt.task_name + '_train.mindrecord'
        if not args_opt.eval_data_file_path:
            args_opt.eval_data_file_path = local_data_path + '/' + args_opt.task_name + '_test.mindrecord'
    print_args_input(args_opt)
    eval_onnx(args_opt)
