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
"""create mindrecord for training EfficientDet."""
import argparse
import os
from mindspore.mindrecord import FileWriter
from src.dataset import create_mindrecord, create_coco_label
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientDet dataset create")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--data_url", type=str, default=None, help="coco raw file obs path")
    parser.add_argument("--train_url", type=str, default=None, help="ckpt output dir in obs")
    parser.add_argument("--is_modelarts", type=str, default="False", help="")

    args_opt = parser.parse_args()

    device_id = int(os.getenv('DEVICE_ID'))

    if args_opt.is_modelarts == "True":
        import moxing as mox
        local_data_url = "/cache/data/" + str(device_id)
        mox.file.make_dirs(local_data_url)
        mindrecord_dir = "/cache/mr/" + str(device_id)
        mox.file.make_dirs(mindrecord_dir)
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        prefix = "EfficientDet.mindrecord"
        mindrecord_path = os.path.join(mindrecord_dir, prefix)
        writer = FileWriter(mindrecord_path, 8)

        images, image_path_dict, image_anno_dict = create_coco_label(True, local_data_url)

        EfficientDet_json = {
            "image": {"type": "bytes"},
            "annotation": {"type": "float32", "shape": [-1, 5]},
        }
        writer.add_schema(EfficientDet_json, "EfficientDet_json")

        for img_id in images:
            image_path = image_path_dict[img_id]
            with open(image_path, 'rb') as f:
                img = f.read()
            annos = np.array(image_anno_dict[img_id], dtype=np.int32)
            img_id = np.array([img_id], dtype=np.int32)
            row = {"img_id": img_id, "image": img, "annotation": annos}
            writer.write_raw_data([row])
        writer.commit()
        print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        mox.file.copy_parallel(mindrecord_dir, args_opt.train_url)
        print("transfer OK. ")

    else:

        mindrecord_file = create_mindrecord("coco", "EfficientDet.mindrecord", True)
