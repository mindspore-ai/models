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

"""Evaluation for SSD"""

import os
import time
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.ssd import SsdInferWithDecoder, ssd_mobilenet_v2_fpn
from src.dataset import create_ssd_dataset, create_mindrecord
from src.eval_utils import metrics
from src.box_utils import default_boxes
from src.model_utils.config import config as cfg

def ssd_eval(dataset_path, ckpt_path, anno_json):
    """SSD evaluation."""
    batch_size = 1
    ds = create_ssd_dataset(dataset_path, batch_size=batch_size, repeat_num=1,
                            is_training=False, use_multiprocessing=False)
    net = ssd_mobilenet_v2_fpn(config=cfg)
    net = SsdInferWithDecoder(net, Tensor(default_boxes), cfg)

    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    i = batch_size
    total = ds.get_dataset_size() * batch_size
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(Tensor(img_np))
        for batch_idx in range(img_np.shape[0]):
            pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                              "box_scores": output[1].asnumpy()[batch_idx],
                              "img_id": int(np.squeeze(img_id[batch_idx])),
                              "image_shape": image_shape[batch_idx]})
        percent = round(i / total * 100., 2)

        print(f'    {str(percent)} [{i}/{total}]', end='\r')
        i += batch_size
    cost_time = int((time.time() - start) * 1000)
    print(f'    100% [{total}/{total}] cost {cost_time} ms')
    mAP = metrics(pred_data, anno_json)
    print("\n========================================\n")
    print(f"mAP: {mAP}")


if __name__ == '__main__':
    if cfg.modelarts_mode:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg.run_platform, device_id=device_id)
        cfg.coco_root = os.path.join(cfg.coco_root, str(device_id))
        cfg.mindrecord_dir = os.path.join(cfg.mindrecord_dir, str(device_id))
        checkpoint_path = "/cache/ckpt/"
        checkpoint_path = os.path.join(checkpoint_path, str(device_id))
        mox.file.copy_parallel(cfg.checkpoint_path, checkpoint_path)
        if cfg.mindrecord_mode == "mindrecord":
            mox.file.copy_parallel(cfg.data_url, cfg.mindrecord_dir)
        else:
            mox.file.copy_parallel(cfg.data_url, cfg.coco_root)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg.run_platform)
        if cfg.run_platform == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)

    mindrecord_file = create_mindrecord(cfg.dataset, "ssd_eval.mindrecord", False)

    if cfg.dataset == "coco":
        json_path = os.path.join(cfg.coco_root, cfg.instances_set.format(cfg.val_data_type))
    elif cfg.dataset == "voc":
        json_path = os.path.join(cfg.voc_root, cfg.voc_json)
    else:
        raise ValueError('SSD eval only support dataset mode is coco and voc!')
    print("Start Eval!")
    if cfg.modelarts_mode:
        checkpoint_path = checkpoint_path + '/ssd-500_458.ckpt'
        ssd_eval(mindrecord_file, checkpoint_path, json_path)
        mox.file.copy_parallel(cfg.mindrecord_dir, cfg.train_url)
    else:
        ssd_eval(mindrecord_file, cfg.checkpoint_path, json_path)
