# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Face attribute eval."""
import os
import time
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype

from src.dataset_eval import data_generator_eval
from src.FaceAttribute.resnet18 import get_resnet18

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


def softmax(x, axis=0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def load_pretrain(checkpoint, network):
    '''load pretrain model.'''
    if os.path.isfile(checkpoint):
        param_dict = load_checkpoint(checkpoint)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('-----------------------load model success-----------------------')
    else:
        print('-----------------------load model failed-----------------------')
    return network


def modelarts_pre_process():
    '''modelarts pre process function.'''

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))


def eval_func(eval_network, eval_dataloader):
    total_data_num_age, total_data_num_gen, total_data_num_mask = 0, 0, 0
    age_num, gen_num, mask_num = 0, 0, 0
    gen_tp_num, mask_tp_num, gen_fp_num = 0, 0, 0
    mask_fp_num, gen_fn_num, mask_fn_num = 0, 0, 0
    for data, gt_classes in eval_dataloader:
        data_tensor = Tensor(data, dtype=mstype.float32)
        fea = eval_network(data_tensor)

        gt_age, gt_gen, gt_mask = gt_classes[0]

        age_result, gen_result, mask_result = fea

        age_result_np = age_result.asnumpy()
        gen_result_np = gen_result.asnumpy()
        mask_result_np = mask_result.asnumpy()

        age_prob = softmax(age_result_np[0].astype(np.float32)).tolist()
        gen_prob = softmax(gen_result_np[0].astype(np.float32)).tolist()
        mask_prob = softmax(mask_result_np[0].astype(np.float32)).tolist()

        age = age_prob.index(max(age_prob))
        gen = gen_prob.index(max(gen_prob))
        mask = mask_prob.index(max(mask_prob))

        age_num += (gt_age == age)
        gen_num += (gt_gen == gen)
        mask_num += (gt_mask == mask)

        gen_tp_num += (gen == 1 and gt_gen == 1)
        gen_fp_num += (gen == 1 and gt_gen == 0)
        gen_fn_num += (gen == 0 and gt_gen == 1)

        mask_tp_num += (gt_mask == 1 and mask == 1)
        mask_fp_num += (gt_mask == 0 and mask == 1)
        mask_fn_num += (gt_mask == 1 and mask == 0)

        total_data_num_age += (gt_age != -1)
        total_data_num_gen += (gt_gen != -1)
        total_data_num_mask += (gt_mask != -1)

    age_accuracy = float(age_num) / float(total_data_num_age)
    gen_precision, gen_recall, gen_accuracy = 0, 0, 0
    if gen_tp_num == 0 and gen_tp_num == 0:
        gen_precision, gen_recall = 0, 0
    else:
        gen_precision = float(gen_tp_num) / (float(gen_tp_num) + float(gen_fp_num))
        gen_recall = float(gen_tp_num) / (float(gen_tp_num) + float(gen_fn_num))
    if gen_precision == 0 and gen_recall == 0:
        gen_f1 = 0
    else:
        gen_f1 = 2. * gen_precision * gen_recall / (gen_precision + gen_recall)
    gen_accuracy = float(gen_num) / float(total_data_num_gen)

    if mask_tp_num == 0 and mask_fp_num == 0:
        mask_precision, mask_recall = 0, 0
    else:
        mask_precision = float(mask_tp_num) / (float(mask_tp_num) + float(mask_fp_num))
        mask_recall = float(mask_tp_num) / (float(mask_tp_num) + float(mask_fn_num))
    if mask_precision == 0 and mask_recall == 0:
        mask_f1 = 0
    else:
        mask_f1 = 2. * mask_precision * mask_recall / (mask_precision + mask_recall)
    mask_accuracy = float(mask_num) / float(total_data_num_mask)

    return total_data_num_age, total_data_num_gen, total_data_num_mask, age_accuracy, \
        gen_accuracy, mask_accuracy, gen_precision, gen_recall, gen_f1, mask_precision, \
        mask_recall, mask_f1


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''run eval.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                        device_id=get_device_id())

    network = get_resnet18(config)
    de_dataloader, _, _ = data_generator_eval(config)
    ckpt_files = os.listdir(config.ckpt_dir)

    best_multiply_accuracy = 0
    for ckpt_file in ckpt_files:
        if not ckpt_file.endswith(".ckpt"):
            continue
        ckpt_path = os.path.join(config.ckpt_dir, ckpt_file)
        network = load_pretrain(ckpt_path, network)
        network.set_train(False)
        result = eval_func(network, de_dataloader)

        if result[3] * result[4] * result[5] > best_multiply_accuracy:
            best_multiply_accuracy = result[3] * result[4] * result[5]
            print("===============================", flush=True)
            print("current best ckpt_path is", ckpt_path, flush=True)
            print('total age num: ', result[0], flush=True)
            print('total gen num: ', result[1], flush=True)
            print('total mask num: ', result[2], flush=True)
            print('age accuracy: ', result[3], flush=True)
            print('gen accuracy: ', result[4], flush=True)
            print('mask accuracy: ', result[5], flush=True)
            print('gen precision: ', result[6], flush=True)
            print('gen recall: ', result[7], flush=True)
            print('gen f1: ', result[8], flush=True)
            print('mask precision: ', result[9], flush=True)
            print('mask recall: ', result[10], flush=True)
            print('mask f1: ', result[11], flush=True)


if __name__ == '__main__':
    run_eval()
