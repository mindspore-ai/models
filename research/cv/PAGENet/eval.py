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
import time
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import context
import config
from config import MODE, device_target, train_size

from src.pagenet import MindsporeModel
from src.mind_dataloader_final import get_test_loader


def main(test_img_path, test_gt_path, ckpt_file):
    # context_set
    context.set_context(mode=MODE,
                        device_target=device_target,
                        reserve_class_name_in_scope=False)

    # dataset
    test_loader = get_test_loader(test_img_path, test_gt_path, batchsize=1, testsize=train_size)
    data_iterator = test_loader.create_tuple_iterator()
    # step
    total_test_step = 0
    test_data_size = test_loader.get_dataset_size()
    # loss&eval
    loss = nn.Loss()
    mae = nn.MAE()
    F_score = nn.F1()
    # model
    model = MindsporeModel()
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
    print("total time is {}h".format(total / 3600))
    print("step time is {}s".format(total / (test_data_size)))
    mae_result = mae.eval()

    F_score_result = F_score.eval()
    print("mae: ", mae_result)

    print("F-score: ", (F_score_result[0] + F_score_result[1]) / 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-s', '--test_set', type=str)
    parser.add_argument('-c', '--ckpt', type=str)
    args = parser.parse_args()
    if args.test_set == 'DUT-OMRON':
        img_path = config.DUT_OMRON_img_path
        gt_path = config.DUT_OMRON_gt_path
    elif args.test_set == 'DUTS-TE':
        img_path = config.DUTS_TE_img_path
        gt_path = config.DUTS_TE_gt_path
    elif args.test_set == 'ECCSD':
        img_path = config.ECCSD_img_path
        gt_path = config.ECCSD_gt_path
    elif args.test_set == 'HKU-IS':
        img_path = config.HKU_IS_img_path
        gt_path = config.HKU_IS_gt_path
    elif args.test_set == 'SOD':
        img_path = config.SOD_img_path
        gt_path = config.SOD_gt_path
    else:
        print("dataset is not exist")
    ckpt = args.ckpt
    main(img_path, gt_path, ckpt)
