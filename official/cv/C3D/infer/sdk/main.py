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
import os
import sys
import csv
import json
import cv2
from PIL import Image
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector
import numpy as np
import MxpiDataType_pb2 as MxpiDataType


def logger():
    """create log path"""
    if not os.path.exists('./result'):
        os.makedirs('./result')
    filePath = './result/accLogData.csv'
    if os.path.exists(filePath):
        os.remove(filePath)
    with open(filePath, mode='w', encoding='utf-8'):
        print("Logger Ready")


def log(filePath, label, predInd):
    """write log"""
    with open(filePath + "/infer_result.csv", 'a', newline='') as csvFile:
        writter = csv.writer(csvFile)
        writter.writerow([label, predInd])
        csvFile.close()


def get_result(stream_name_, stream_manager, out_plugin_id=0):
    """get result"""
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager.GetProtobuf(
        stream_name_, out_plugin_id, key_vec)
    if infer_result.size() == 0:
        print("inferResult is null")
        return None
    result_ = MxpiDataType.MxpiTensorPackageList()
    result_.ParseFromString(infer_result[0].messageBuf)
    return np.frombuffer(result_.tensorPackageVec[0].tensorVec[0].dataStr,
                         dtype=np.float32)


def send_protobuf(stream_name_, plugin_id, pkg_list, stream_manager):
    """" send data buffer to om """
    protobuf = MxProtobufIn()
    protobuf.key = "appsrc{}".format(plugin_id).encode('utf-8')
    protobuf.type = b"MxTools.MxpiTensorPackageList"
    protobuf.protobuf = pkg_list.SerializeToString()
    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf)
    stream_manager.SendProtobuf(stream_name_, plugin_id, protobuf_vec)
    return True


def send_tensor_input(stream_name_, plugin_id,
                      input_data, input_shape, stream_manager):
    """" send tensor data to om """
    tensor_list = MxpiDataType.MxpiTensorPackageList()
    tensor_pkg = tensor_list.tensorPackageVec.add()
    # init tensor vector
    tensor_vec = tensor_pkg.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    tensor_vec.tensorShape.extend(input_shape)
    tensor_vec.tensorDataType = 0
    tensor_vec.dataStr = input_data
    tensor_vec.tensorDataSize = len(input_data)
    return send_protobuf(stream_name_, plugin_id, tensor_list, stream_manager)


class PreprocessEvalC3D():
    """
    Container for all transforms used to preprocess clips.
    """

    def __init__(self):
        """
        Initialize preprocessing class for training set
        """
        self.transforms = []
        self.clip_mean = np.load(
            "../../pretrained_model/sport1m_train16_128_mean.npy")[0]
        self.clip_mean = np.transpose(self.clip_mean, (1, 2, 3, 0))
        self.transforms.append(ResizeClip())
        self.transforms.append(SubtractMeanClip(
            clip_mean=self.clip_mean))
        self.transforms.append(CenterCropClip())

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)
        return input_data


class ResizeClip():
    """resize clip pictures into the input size"""

    def __init__(self):
        self.size_h, self.size_w = 128, 171  # 128 171

    def resize_bbox(self, xmin, ymin, xmax, ymax, img_shape, resize_shape):
        """
        Resize a bounding box within a frame relative
        to the amountthat the frame was resized
        """
        img_h = img_shape[0]
        img_w = img_shape[1]

        res_h = resize_shape[0]
        res_w = resize_shape[1]

        frac_h = res_h / float(img_h)
        frac_w = res_w / float(img_w)

        xmin_new = int(xmin * frac_w)
        xmax_new = int(xmax * frac_w)

        ymin_new = int(ymin * frac_h)
        ymax_new = int(ymax * frac_h)

        return xmin_new, ymin_new, xmax_new, ymax_new

    def resize_pt_coords(self, x, y, img_shape, resize_shape):
        """
        Get relative position for point coords
        within a frame after it's resized
        """

        img_h = img_shape[0]
        img_w = img_shape[1]

        res_h = resize_shape[0]
        res_w = resize_shape[1]

        frac_h = res_h / float(img_h)
        frac_w = res_w / float(img_w)

        x_new = (x * frac_w).astype(int)
        y_new = (y * frac_h).astype(int)

        return x_new, y_new

    def __call__(self, clip, bbox=None):

        clip = self._to_numpy(clip)
        out_clip = []
        out_bbox = []
        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]

            proc_frame = cv2.resize(frame, (self.size_w, self.size_h))
            out_clip.append(proc_frame)
            if bbox:
                temp_bbox = np.zeros(bbox[frame_ind].shape) - 1
                for class_ind, box in enumerate(bbox[frame_ind]):
                    if np.array_equal(box, -1 * np.ones(box.shape)):
                        continue

                    if box.shape[-1] == 2:  # Operate on point coordinates
                        proc_bbox = np.stack(
                            self.resize_pt_coords(box[:, 0], box[:, 1],
                                                  frame.shape, (self.size_h, self.size_w)), 1)
                    else:  # Operate on bounding box
                        xmin, ymin, xmax, ymax = box
                        proc_bbox = self.resize_bbox(
                            xmin, ymin, xmax, ymax, frame.shape,
                            (self.size_h, self.size_w))

                    temp_bbox[class_ind, :] = proc_bbox
                out_bbox.append(temp_bbox)

        out_clip = np.array(out_clip)

        assert (out_clip.shape[1:3] == (self.size_h, self.size_w)), \
            'Proc frame: {} Crop h,w: {},{}'.format(out_clip.shape, self.size_h, self.size_w)

        if bbox:
            return out_clip, out_bbox

        return out_clip

    def _to_numpy(self, clip):
        output = []
        if isinstance(clip[0], Image.Image):
            for frame in clip:
                output.append(np.array(frame))
        else:
            output = clip
        output = np.array(output)

        return output


class SubtractMeanClip():
    """preprocess the clip with sport1m mean file"""

    def __init__(self, clip_mean):
        self.clip_mean = clip_mean

    def __call__(self, clip, bbox=None):
        for clip_ind in range(len(clip)):
            clip[clip_ind] = clip[clip_ind] - self.clip_mean[clip_ind]
        if bbox:
            return clip, bbox

        return clip


class CenterCropClip():
    """crop the center of the clip"""

    def __init__(self):
        self.crop_h, self.crop_w = 112, 112  # 112 112
        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

    def _calculate_center(self, frame_h, frame_w):  # 128 171
        xmin = int(frame_w / 2 - self.crop_w / 2)
        xmax = int(frame_w / 2 + self.crop_w / 2)
        ymin = int(frame_h / 2 - self.crop_h / 2)
        ymax = int(frame_h / 2 + self.crop_h / 2)
        return xmin, xmax, ymin, ymax

    def __call__(self, clip, bbox=None):
        frame_shape = clip[0].shape
        xmin, xmax, ymin, ymax = self._calculate_center(
            frame_shape[0], frame_shape[1])
        self.crop_transform.update_bbox(xmin, xmax, ymin, ymax)
        proc_clip = self.crop_transform(clip, bbox)
        if isinstance(proc_clip, tuple):
            assert (proc_clip[0].shape[1:3] == (self.crop_h, self.crop_w)), \
                'Proc frame: {} Crop h,w: {},{}'.format(proc_clip[0].shape, self.crop_h, self.crop_w)
        else:
            assert (proc_clip.shape[1:3] == (self.crop_h, self.crop_w)), \
                'Proc frame: {} Crop h,w: {},{}'.format(proc_clip.shape, self.crop_h, self.crop_w)
        return proc_clip


class CropClip():
    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None):
        self.crop_xmin = xmin
        self.crop_xmax = xmax
        self.crop_ymin = ymin
        self.crop_ymax = ymax

        self.crop_h, self.crop_w = 112, 112

    def update_bbox(self, xmin, xmax, ymin, ymax, update_crop_shape=False):
        '''
            Args:
                xmin (Float, shape []):
                xmax (Float, shape []):
                ymin (Float, shape []):
                ymax (Float, shape []):
                update_crop_shape (Boolean): Update expected crop shape
                along with bbox update call
        '''
        self.crop_xmin = xmin
        self.crop_xmax = xmax
        self.crop_ymin = ymin
        self.crop_ymax = ymax

        if update_crop_shape:
            self.crop_h = ymax - ymin
            self.crop_w = xmax - xmin

    def crop_bbox(self, xmin, ymin, xmax, ymax, crop_xmin,
                  crop_ymin, crop_xmax, crop_ymax):
        if (xmin >= crop_xmax) or (xmax <= crop_xmin) or \
                (ymin >= crop_ymax) or (ymax <= crop_ymin):
            return -1, -1, -1, -1

        if ymax > crop_ymax:
            ymax_new = crop_ymax
        else:
            ymax_new = ymax

        if xmax > crop_xmax:
            xmax_new = crop_xmax
        else:
            xmax_new = xmax

        if ymin < crop_ymin:
            ymin_new = crop_ymin
        else:
            ymin_new = ymin

        if xmin < crop_xmin:
            xmin_new = crop_xmin
        else:
            xmin_new = xmin

        return xmin_new - crop_xmin, ymin_new - crop_ymin, \
            xmax_new - crop_xmin, ymax_new - crop_ymin

    def crop_coords(self, x, y, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        if np.any(x >= crop_xmax) or np.any(x <= crop_xmin) or \
                np.any(y >= crop_ymax) or np.any(y <= crop_ymin):
            return -1 * np.ones(x.shape), -1 * np.ones(y.shape)

        x_new = np.clip(x, crop_xmin, crop_xmax)
        y_new = np.clip(y, crop_ymin, crop_ymax)

        return x_new - crop_xmin, y_new - crop_ymin

    def __call__(self, clip, bbox=None):
        out_clip = []
        out_bbox = []

        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            proc_frame = np.array(
                frame[self.crop_ymin:self.crop_ymax,
                      self.crop_xmin:self.crop_xmax])
            out_clip.append(proc_frame)

            assert (proc_frame.shape[:2] == (self.crop_h, self.crop_w)), \
                'Frame shape: {}, Proc frame: {} Crop h,w: {},{}'.format(
                    frame.shape, proc_frame.shape, self.crop_h, self.crop_w)

            if bbox:
                temp_bbox = np.zeros(bbox[frame_ind].shape) - 1
                for class_ind, box in enumerate(bbox[frame_ind]):
                    if np.array_equal(box, -1 * np.ones(box.shape)):
                        continue

                    if box.shape[-1] == 2:  # Operate on point coordinates
                        proc_bbox = np.stack(self.crop_coords(
                            box[:, 0], box[:, 1], self.crop_xmin, self.crop_ymin, self.crop_xmax, self.crop_ymax), 1)
                    else:  # Operate on bounding box
                        xmin, ymin, xmax, ymax = box
                        proc_bbox = self.crop_bbox(xmin, ymin, xmax, ymax, self.crop_xmin, self.crop_ymin,
                                                   self.crop_xmax, self.crop_ymax)
                    temp_bbox[class_ind, :] = proc_bbox
                out_bbox.append(temp_bbox)

        if bbox:
            return np.array(out_clip), np.array(out_bbox)

        return np.array(out_clip)


class Dataset():
    """the UCF101 dataset class to get and preprocess the dataset """

    def __init__(self):
        """
        get Samples
        """
        self.img_path = sys.argv[1]
        self.getSamples(sys.argv[2])
        self.transforms = PreprocessEvalC3D()

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        base_path = os.path.join(self.img_path, vid_info['base_path'])
        vid_length = len(vid_info['frames'])
        labels = np.zeros((vid_length)) - 1
        input_data = []
        for frame_ind in range(len(vid_info['frames'])):
            frame_path = os.path.join(
                base_path, vid_info['frames'][frame_ind]['img_path'])

            for frame_labels in vid_info['frames'][frame_ind]['actions']:
                labels[frame_ind] = frame_labels['action_class']
            # Load frame image data and preprocess image accordingly
            input_data.append(cv2.imread(frame_path)[..., ::-1] / 1.)
        # Preprocess data
        vid_data = self.transforms(input_data)
        vid_data = np.transpose(vid_data, (3, 0, 1, 2))
        return (vid_data.astype(np.float32), labels.astype(np.int32))

    def getSamples(self, jsonFilePath):
        jsonFile = open(jsonFilePath, 'r')
        jsonData = json.load(jsonFile)
        jsonFile.close()
        self.samples = []
        for videoInfo in jsonData:
            clips = self.extractClips(videoInfo['frames'])
            for clip in clips:
                self.samples.append(
                    dict(frames=clip, base_path=videoInfo['base_path']))

    def extractClips(self, video):
        if len(video) >= 16:
            final_video = [video[_idx] for _idx in np.linspace(
                0, len(video) - 1, 16, dtype='int32')]
            final_video = [final_video]

        else:
            indices = np.ceil(16 / float(len(video)))
            indices = indices.astype('int32')
            indices = np.tile(
                np.arange(0, len(video), 1, dtype='int32'), indices)
            indices = indices[np.linspace(
                0, len(indices) - 1, 16, dtype='int32')]

            final_video = [video[_idx] for _idx in indices]
            final_video = [final_video]
        return final_video


def run():
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline file
    with open("./c3d.pipeline", 'rb') as f:
        pipeline = f.read()
        ret = streamManagerApi.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    logger()
    dataset = Dataset()
    acc_sum, sample_num = 0, 0
    for _, (input_data, label) in enumerate(dataset):
        input_data = np.expand_dims(input_data, axis=0)
        send_tensor_input(b"c3d", 0, input_data.tobytes(),
                          input_data.shape, streamManagerApi)
        preds = get_result(b"c3d", streamManagerApi)
        pred = np.argmax(preds, 0)
        sample_num += 1
        if label[0] == pred:
            acc_sum += 1
        if sample_num % 10 == 0:
            accPercent = "%.3f%%" % (acc_sum / sample_num * 100)
            print("current accuracy: {}".format(accPercent))
            log(sys.argv[3], label[0], pred)
        else:
            log(sys.argv[3], label[0], pred)

    streamManagerApi.DestroyAllStreams()


if __name__ == '__main__':
    run()
