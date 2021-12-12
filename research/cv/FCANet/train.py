# Copyright 2021 Huawei Technologies Co., Ltd
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
""" train fcanet """
import os
import time
from src.config import config
from src.trainer import Trainer

if __name__ == "__main__":
    # set config
    p = config
    p["resume"] = None
    p["snapshot_path"] = "./snapshot"
    os.makedirs(p["snapshot_path"], exist_ok=True)
    split_line_num = 79
    # start
    print(
        "Start time : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    )
    print("-" * split_line_num, "\ninfos : ", p, "\n" + "-" * split_line_num)

    # set trainer
    mine = Trainer(p)

    for epoch in range(p["epochs"]):
        lr_str = "{:.7f}".format(mine.scheduler.get_lr())
        print(
            "-" * split_line_num + "\n" + "Epoch [{:03d}]=>    |-lr:{}-|  \n".format(epoch, lr_str)
        )
        # training
        if p["train_only_epochs"] >= 0:
            mine.training(epoch)
            mine.scheduler.step()

        if epoch < p["train_only_epochs"]:
            continue

        # validation-robot
        if (epoch + 1) % p["val_robot_interval"] == 0:
            mine.save_model_ckpt(
                "{}/model-epoch-{}.ckpt".format(p["snapshot_path"], str(epoch).zfill(3))
            )
            mine.validation_robot(epoch, tsh=p["pred_tsh"], resize=p["size"][0])
    print(
        "-" * split_line_num + "\nEnd time : ",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )
