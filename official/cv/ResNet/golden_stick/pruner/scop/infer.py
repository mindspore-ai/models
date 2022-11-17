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
"""infer scop_resnet mindir."""
import datetime
import mindspore as ms
import mindspore.nn as nn
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset


@moxing_wrapper()
def infer_net():
    target = config.device_target
    if target != "GPU":
        raise ValueError("Currently only support GPU.")

    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size, target=target)
    step_size = dataset.get_dataset_size()

    # load mindir
    graph = ms.load(config.mindir_path)
    net = nn.GraphCell(graph)

    print("start infer")
    total_time = 0
    data_loader = dataset.create_dict_iterator(num_epochs=1)
    for _, data in enumerate(data_loader):
        images = data["image"]
        start_time = datetime.datetime.now()
        net(ms.Tensor(images))
        end_time = datetime.datetime.now()
        total_time += (end_time - start_time).microseconds

    step_time = total_time / step_size
    print("per step time: {} ms".format(step_time))

if __name__ == '__main__':
    infer_net()
