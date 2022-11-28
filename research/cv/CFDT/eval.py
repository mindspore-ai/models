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
"""eval"""
import os
from tqdm import tqdm
import mindspore as ms
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from src.dataset.coco_eval import post_process, CocoEvaluator
from src.dataset import build_dataset
from src.config import config
from src.cfdt import build_cfdt

def run_eval():
    """run evaluate"""
    ms.set_seed(config.seed)
    config.device_num = 1
    config.rank = 0
    # for evaluate, set aux_loss to false
    config.aux_loss = False
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=device_id)
    cfdt = build_cfdt(config)
    load_param_into_net(cfdt, load_checkpoint(config.ckpt_path))
    cfdt.set_train(False)
    dataset, base_ds = build_dataset(config)
    data_loader = dataset.create_dict_iterator()
    coco_evaluator = CocoEvaluator(base_ds)
    print('Start evaluation', flush=True)
    for sample in tqdm(data_loader):

        images = sample['image']
        masks = sample['mask']
        outputs = cfdt(images, masks)
        outputs = {'pred_logits': outputs[:, :, :91],
                   'pred_boxes': outputs[:, :, 91:]}

        # compute AP, etc
        orig_target_sizes = sample['orig_sizes'].asnumpy()
        results = post_process(outputs, orig_target_sizes)
        res = {img_id: output for img_id, output in zip(sample['img_id'].asnumpy(), results)}
        coco_evaluator.update(res)
    coco_evaluator.write_result()
    eval_result = coco_evaluator.get_eval_result()
    print(eval_result)


if __name__ == "__main__":
    run_eval()
