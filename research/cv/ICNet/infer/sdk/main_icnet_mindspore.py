# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""the main sdk infer file"""
import argparse
import os

from PIL import Image
import numpy as np
from StreamManagerApi import MxDataInput, StringVector, StreamManagerApi
import MxpiDataType_pb2 as MxpiDataType

PIPELINE_PATH = "./config/icnet_mindspore.pipeline"
INFER_RESULT_DIR = "./result"


def _parse_args():
    parser = argparse.ArgumentParser('mindspore icnet eval')
    parser.add_argument('--data_root', type=str, default='',
                        help='root path of val data')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='number of classes')
    args, _ = parser.parse_known_args()
    return args


def _get_val_pairs(folder):
    """get val img_mask_path_pairs"""
    split = 'val'
    image_folder = os.path.join(folder, 'leftImg8bit/' + split)
    mask_folder = os.path.join(folder, 'gtFine/' + split)
    img_paths = []
    mask_paths = []
    for root, _, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith('.png'):
                imgpath = os.path.join(root, filename)
                foldername = os.path.basename(os.path.dirname(imgpath))
                maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                maskpath = os.path.join(mask_folder, foldername, maskname)
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)
    print('Found {} images in the folder {}'.format(len(img_paths), image_folder))
    return img_paths, mask_paths


def _do_infer(stream_manager_api, data_input):
    """
    send images into stream to do infer

    Returns:
        infer result, numpy array
    """
    stream_name = b'segmentation'
    unique_id = stream_manager_api.SendData(
        stream_name, 0, data_input)
    if unique_id < 0:
        raise RuntimeError("Failed to send data to stream.")

    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)
    print(infer_result)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    TensorList = MxpiDataType.MxpiTensorPackageList()
    TensorList.ParseFromString(infer_result[0].messageBuf)
    data = np.frombuffer(
        TensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    data = data.reshape(1, 19, 1024, 2048)
    return data


def _class_to_index(mask):
    """assert the value"""
    values = np.unique(mask)
    _key = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1,
                     2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18])
    _mapping = np.array(range(-1, len(_key) - 1)).astype('int32')
    for value in values:
        assert value in _mapping
    index = np.digitize(mask.ravel(), _mapping, right=True)
    return _key[index].reshape(mask.shape)


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    predict = np.argmax(output, axis=1) + 1
    target = target.astype(float) + 1

    predict = predict.astype(float) * np.array(target > 0).astype(float)
    intersection = predict * np.array(predict == target).astype(float)

    area_inter, _ = np.array(np.histogram(intersection, bins=nclass, range=(1, nclass+1)))
    area_pred, _ = np.array(np.histogram(predict, bins=nclass, range=(1, nclass+1)))
    area_lab, _ = np.array(np.histogram(target, bins=nclass, range=(1, nclass+1)))

    area_all = area_pred + area_lab
    area_union = area_all - area_inter

    return area_inter, area_union


cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def main():
    args = _parse_args()

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open(PIPELINE_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    os.makedirs(INFER_RESULT_DIR, exist_ok=True)

    data_input = MxDataInput()
    total_inter = np.zeros(args.num_classes, dtype=np.float32)
    total_union = np.zeros(args.num_classes, dtype=np.float32)
    image_paths, mask_paths = _get_val_pairs(args.data_root)
    for i in range(len(image_paths)):
        print("img_path:", image_paths[i])
        mask = Image.open(mask_paths[i])
        mask = _class_to_index(np.array(mask).astype('int32'))
        with open(image_paths[i], 'rb') as f:
            data_input.data = f.read()
        each_array = _do_infer(stream_manager_api, data_input)

        mask = np.expand_dims(mask, axis=0)

        pred = np.argmax(each_array, axis=1)
        pred = pred.squeeze(0)
        out_img = Image.fromarray(pred.astype('uint8'))
        out_img.putpalette(cityspallete)
        result_path = os.path.join(
            INFER_RESULT_DIR,
            f"{image_paths[i].split('/')[-1].split('.')[0]}sdk_infer.png")
        out_img.save(result_path)

        inter, union = batch_intersection_union(each_array, mask, args.num_classes)

        total_inter = inter + total_inter
        total_union = union + total_union
    Iou = np.true_divide(total_inter, (2.220446049250313e-16 + total_union))
    print("mean IoU", np.mean(Iou))

    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
