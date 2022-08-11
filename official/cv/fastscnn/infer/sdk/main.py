'''
The scripts to execute sdk infer
'''
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

import argparse
import os
import time
import numpy as np
import PIL.Image as Image
from tabulate import tabulate

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="FastSCNN process")
    parser.add_argument("--pipeline", type=str, default=None, help="SDK infer pipeline")
    parser.add_argument("--image_path", type=str, default=None, help="root path of image")
    parser.add_argument('--image_width', default=768, type=int, help='image width')
    parser.add_argument('--image_height', default=768, type=int, help='image height')
    parser.add_argument('--save_mask', default=1, type=int, help='0 for False, 1 for True')
    parser.add_argument('--mask_result_path', default='./mask_result', type=str,
                        help='the folder to save the semantic mask images')
    args_opt = parser.parse_args()
    return args_opt

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True

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
classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')

valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 31, 32, 33]

_key = np.array([-1, -1, -1, -1, -1, -1,
                 -1, -1, 0, 1, -1, -1,
                 2, 3, 4, -1, -1, -1,
                 5, -1, 6, 7, 8, 9,
                 10, 11, 12, 13, 14, 15,
                 -1, -1, 16, 17, 18])
_mapping = np.array(range(-1, len(_key) - 1)).astype('int32')

def _get_city_pairs(folder, split='train'):
    '''_get_city_pairs'''
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
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
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + split)
        mask_folder = os.path.join(folder, 'gtFine' + os.sep + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    assert split == 'trainval'
    print('trainval set')
    train_img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + 'train')
    train_mask_folder = os.path.join(folder, 'gtFine' + os.sep + 'train')
    val_img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + 'val')
    val_mask_folder = os.path.join(folder, 'gtFine' + os.sep + 'val')
    train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
    val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
    img_paths = train_img_paths + val_img_paths
    mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths

def _val_sync_transform(outsize, img, mask):
    '''_val_sync_transform'''
    short_size = min(outsize)
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize[1]) / 2.))
    y1 = int(round((h - outsize[0]) / 2.))
    img = img.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
    mask = mask.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))

    # final transform
    img, mask = np.array(img), _mask_transform(mask)
    return img, mask

def _class_to_index(mask):
    # assert the value
    values = np.unique(mask)
    for value in values:
        assert value in _mapping
    index = np.digitize(mask.ravel(), _mapping, right=True)
    return _key[index].reshape(mask.shape)

def _mask_transform(mask):
    target = _class_to_index(np.array(mask).astype('int32'))
    return np.array(target).astype('int32')
class SegmentationMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            self.total_correct += correct
            self.total_label += labeled
            self.total_inter += inter
            self.total_union += union
        evaluate_worker(self, preds, labels)

    def get(self, return_category_iou=False):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        # remove np.spacing(1)
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        if return_category_iou:
            return pixAcc, mIoU, IoU
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.zeros(self.nclass)
        self.total_union = np.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D NCHW where 'C' means label classes, target 3D NHW

    predict = np.argmax(output.astype(np.int64), 1) + 1
    target = target.astype(np.int64) + 1
    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = np.argmax(output.astype(np.float32), 1) + 1
    target = target.astype(np.float32) + 1

    predict = predict.astype(np.float32) * (target > 0).astype(np.float32)
    intersection = predict * (predict == target).astype(np.float32)
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter > area_union).sum() == 0, "Intersection area should be smaller than Union area"
    return area_inter.astype(np.float32), area_union.astype(np.float32)

def main():
    """
    read pipeline and do infer
    """

    args = parse_args()

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'fastscnn'
    infer_total_time = 0
    assert os.path.exists(args.image_path), "Please put dataset in " + str(args.image_path)
    images, mask_paths = _get_city_pairs(args.image_path, 'val')
    assert len(images) == len(mask_paths)
    if not images:
        raise RuntimeError("Found 0 images in subfolders of:" + args.image_path + "\n")

    if args.save_mask and not os.path.exists(args.mask_result_path):
        os.makedirs(args.mask_result_path)
    metric = SegmentationMetric(19)
    metric.reset()
    for index in range(len(images)):
        image_name = images[index].split(os.sep)[-1].split(".")[0]  # get the name of image file
        print("Processing ---> ", image_name)
        img = Image.open(images[index]).convert('RGB')
        mask = Image.open(mask_paths[index])
        img, mask = _val_sync_transform((args.image_height, args.image_width), img, mask)

        img = img.astype(np.float32)
        mask = mask.astype(np.int32)
        # Computed from random subset of ImageNet training images
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img.transpose((2, 0, 1))#HWC->CHW
        for channel, _ in enumerate(img):
            # Normalization
            img[channel] /= 255
            img[channel] -= mean[channel]
            img[channel] /= std[channel]

        img = np.expand_dims(img, 0)#NCHW
        mask = np.expand_dims(mask, 0)#NHW

        if not send_source_data(0, img, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'modelInfer')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        mask_image = res.reshape(1, 19, args.image_height, args.image_width)

        metric.update(mask_image, mask)
        pixAcc, mIoU = metric.get()
        print("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(index + 1, pixAcc * 100, mIoU * 100))
        if args.save_mask:
            output = np.argmax(mask_image[0], axis=0)
            out_img = Image.fromarray(output.astype('uint8'))
            out_img.putpalette(cityspallete)
            outname = str(image_name) + '.png'
            out_img.save(os.path.join(args.mask_result_path, outname))

    pixAcc, mIoU, category_iou = metric.get(return_category_iou=True)
    print('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(pixAcc * 100, mIoU * 100))
    txtName = os.path.join(args.mask_result_path, "eval_results.txt")
    with open(txtName, "w") as f:
        string = 'validation pixAcc:' + str(pixAcc * 100) + ', mIoU:' + str(mIoU * 100)
        f.write(string)
        f.write('\n')
        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(classes):
            table.append([cls_name, category_iou[i]])
            string = 'class name: ' + cls_name + ' iou: ' + str(category_iou[i]) + '\n'
            f.write(string)
        print('Category iou: \n {}'.format(tabulate(table, headers, \
                               tablefmt='grid', showindex="always", numalign='center', stralign='center')))
    print("Testing finished....")
    print("=======================================")
    print("The total time of inference is {} s".format(infer_total_time))
    print("=======================================")

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    main()
