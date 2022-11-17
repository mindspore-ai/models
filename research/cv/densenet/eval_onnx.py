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
"""Run evaluation for a model exported to ONNX"""
import numpy as np
import onnxruntime as ort
from src.model_utils.config import get_config
from tqdm import tqdm

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count

def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_eval(config, checkpoint_path, data_dir,
             target_device):
    session, input_name = create_session(checkpoint_path, target_device)

    if config.dataset == "cifar10":
        from src.datasets import classification_dataset_cifar10 as classification_dataset
    else:
        from src.datasets import classification_dataset_imagenet as classification_dataset
    config.image_size = list(map(int, config.image_size.split(',')))
    de_dataset = classification_dataset(data_dir, image_size=config.image_size,
                                        per_batch_size=config.per_batch_size,
                                        max_epoch=1, rank=config.rank, group_size=config.group_size,
                                        mode='eval')
    eval_dataloader = de_dataset.create_tuple_iterator()
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    print(de_dataset.get_dataset_size())
    for data, gt_classes in tqdm(eval_dataloader):

        output = session.run(None, {input_name: data.asnumpy()})[0]

        gt_classes = gt_classes.asnumpy()

        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[:, -5:]

        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += config.per_batch_size
    results = [[top1_correct], [top5_correct], [img_tot]]
    results = np.array(results)
    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    acc5 = 100.0 * top5_correct / img_tot
    print('after allreduce eval: top1_correct={}, tot={}, acc={:.2f}%'.format(top1_correct, img_tot, acc1))
    if config.dataset == 'imagenet':
        print("after allreduce eval: top5_correct={}, tot={}, acc={:.2f}%".format(top5_correct, img_tot, acc5))



if __name__ == '__main__':
    config1 = get_config()

    run_eval(config1, config1.ckpt_files, config1.eval_data_dir, config1.device_target)
