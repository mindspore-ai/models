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
import numpy as np


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
    dataset = Dataset()
    if not os.path.exists('./data/UCF101csv/'):
        os.makedirs('./data/UCF101csv/')
    base_p = './data/UCF101csv/'
    f2 = open("./data/csvfilename.csv", mode='a')
    writer = csv.writer(f2)
    for index, (input_data, label) in enumerate(dataset):
        for i in range(16):
            img = input_data[:, i, :, :]
            img = np.transpose(img, (1, 2, 0))
            csvpath = base_p + str(label[0]) + '_' + str(index) + 'input' + str(i) +'.csv'
            csvname = str(label[0]) + '_' + str(index) + 'input' + str(i) +'.csv'
            with open(csvpath, "ab") as f:
                for iii in range(3):
                    np.savetxt(f, img[:, :, iii], delimiter=',')
            print('Files ', csvpath, ' Write success.')
            writer.writerow([label[0], csvname, index])
    f2.close()
    print('          ')
    print('MxBase datasets create success.')
    print('          ')


if __name__ == '__main__':
    run()
