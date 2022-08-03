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

import time
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import context

from src.pagenet import MindsporeModel
from src.mind_dataloader_final import get_test_loader
from src.model_utils.config import config


def main(test_img_path, test_gt_path, ckpt_file):
    # context_set
    context.set_context(mode=config.MODE,
                        device_target=config.device_target,
                        reserve_class_name_in_scope=False)

    # dataset
    test_loader = get_test_loader(test_img_path, test_gt_path, batchsize=1, testsize=config.train_size)
    data_iterator = test_loader.create_tuple_iterator()
    # step
    total_test_step = 0
    test_data_size = test_loader.get_dataset_size()
    # loss&eval
    loss = nn.Loss()
    mae = nn.MAE()
    F_score = nn.F1()
    # model
    model = MindsporeModel(config)
    ckpt_file_name = ckpt_file
    ms.load_checkpoint(ckpt_file_name, net=model)

    model.set_train(False)

    mae.clear()
    loss.clear()
    start = time.time()
    for imgs, targets in data_iterator:

        targets1 = targets.astype(int)
        outputs = model(imgs)
        pre_mask = outputs[9]
        pre_mask = pre_mask.flatten()
        targets1 = targets1.flatten()

        pre_mask1 = pre_mask.asnumpy().tolist()

        F_pre = np.array([[1 - i, i] for i in pre_mask1])

        F_score.update(F_pre, targets1)

        mae.update(pre_mask, targets1)

        total_test_step = total_test_step + 1
        if total_test_step % 100 == 0:
            print("evaling:{}/{}".format(total_test_step, test_data_size))

    end = time.time()
    total = end - start
    print(f"task: {config.test_task}")
    print("total time is {}h".format(total / 3600))
    print("step time is {}s".format(total / (test_data_size)))
    mae_result = mae.eval()

    F_score_result = F_score.eval()
    print("mae: ", mae_result)

    print("F-score: ", (F_score_result[0] + F_score_result[1]) / 2)


if __name__ == "__main__":
    main(config.test_img_path, config.test_gt_path, config.ckpt_file)
