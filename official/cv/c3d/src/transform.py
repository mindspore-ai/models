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

import cv2
import mindspore
import numpy as np
from PIL import Image, ImageFile

from src.model_utils.config import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PreprocTransform:
    """
    Abstract class for preprocessing transforms that contains methods to convert clips to PIL images.
    """

    def __init__(self):
        pass

    def _to_numpy(self, clip):
        output = []
        if isinstance(clip[0], mindspore.Tensor):
            if isinstance(clip, mindspore.Tensor):
                output = clip.numpy()
            else:
                for frame in clip:
                    f_shape = frame.shape
                    # Convert from torch's C, H, W to numpy H, W, C
                    frame = frame.numpy().reshape(f_shape[1], f_shape[2], f_shape[0])
                    output.append(frame)
        elif isinstance(clip[0], Image.Image):
            for frame in clip:
                output.append(np.array(frame))
        else:
            output = clip
        output = np.array(output)

        return output

    def _to_tensor(self, clip):
        """
        torchvision converts PIL images and numpy arrays that are uint8 0 to 255 to float 0 to 1
        Converts numpy arrays that are float to float tensor
        """
        if isinstance(clip[0], mindspore.Tensor):
            return clip

        output = []
        for frame in clip:
            output.append(mindspore.Tensor.from_numpy(frame))

        return output


class ResizeClip(PreprocTransform):

    def __init__(self):
        super(ResizeClip, self).__init__()
        self.size_h, self.size_w = config.resize_shape

    def resize_bbox(self, xmin, ymin, xmax, ymax, img_shape, resize_shape):
        # Resize a bounding box within a frame relative to the amount that the frame was resized

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
        # Get relative position for point coords within a frame, after it's resized

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
                    if np.array_equal(box, -1 * np.ones(box.shape)):  # only annotated objects
                        continue

                    if box.shape[-1] == 2:  # Operate on point coordinates
                        proc_bbox = np.stack(
                            self.resize_pt_coords(box[:, 0], box[:, 1], frame.shape, (self.size_h, self.size_w)), 1)
                    else:  # Operate on bounding box
                        xmin, ymin, xmax, ymax = box
                        proc_bbox = self.resize_bbox(xmin, ymin, xmax, ymax, frame.shape, (self.size_h, self.size_w))

                    temp_bbox[class_ind, :] = proc_bbox
                out_bbox.append(temp_bbox)

        out_clip = np.array(out_clip)

        assert (out_clip.shape[1:3] == (self.size_h, self.size_w)), 'Proc frame: {} Crop h,w: {},{}'.format(
            out_clip.shape, self.size_h, self.size_w)

        if bbox:
            return out_clip, out_bbox

        return out_clip



class SubtractMeanClip(PreprocTransform):
    def __init__(self, clip_mean):
        super(SubtractMeanClip, self).__init__()
        self.clip_mean = clip_mean

    def __call__(self, clip, bbox=None):
        # clip = clip-self.clip_mean
        for clip_ind in range(len(clip)):
            clip[clip_ind] = clip[clip_ind] - self.clip_mean[clip_ind]

        if bbox:
            return clip, bbox

        return clip


class CropClip(PreprocTransform):
    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None):
        super(CropClip, self).__init__()
        self.crop_xmin = xmin
        self.crop_xmax = xmax
        self.crop_ymin = ymin
        self.crop_ymax = ymax

        self.crop_h, self.crop_w = config.crop_shape

    def update_bbox(self, xmin, xmax, ymin, ymax, update_crop_shape=False):
        '''
            Args:
                xmin (Float, shape []):
                xmax (Float, shape []):
                ymin (Float, shape []):
                ymax (Float, shape []):
                update_crop_shape (Boolean): Update expected crop shape along with bbox update call
        '''
        self.crop_xmin = xmin
        self.crop_xmax = xmax
        self.crop_ymin = ymin
        self.crop_ymax = ymax

        if update_crop_shape:
            self.crop_h = ymax - ymin
            self.crop_w = xmax - xmin

    def crop_bbox(self, xmin, ymin, xmax, ymax, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        if (xmin >= crop_xmax) or (xmax <= crop_xmin) or (ymin >= crop_ymax) or (ymax <= crop_ymin):
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

        return xmin_new - crop_xmin, ymin_new - crop_ymin, xmax_new - crop_xmin, ymax_new - crop_ymin

    def crop_coords(self, x, y, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        if np.any(x >= crop_xmax) or np.any(x <= crop_xmin) or np.any(y >= crop_ymax) or np.any(y <= crop_ymin):
            return -1 * np.ones(x.shape), -1 * np.ones(y.shape)

        x_new = np.clip(x, crop_xmin, crop_xmax)
        y_new = np.clip(y, crop_ymin, crop_ymax)

        return x_new - crop_xmin, y_new - crop_ymin

    def __call__(self, clip, bbox=None):
        out_clip = []
        out_bbox = []

        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            proc_frame = np.array(frame[self.crop_ymin:self.crop_ymax, self.crop_xmin:self.crop_xmax])
            out_clip.append(proc_frame)

            assert (proc_frame.shape[:2] == (
                self.crop_h, self.crop_w)), 'Frame shape: {}, Proc frame: {} Crop h,w: {},{}'.format \
                (frame.shape, proc_frame.shape, self.crop_h, self.crop_w)

            if bbox:
                temp_bbox = np.zeros(bbox[frame_ind].shape) - 1
                for class_ind, box in enumerate(bbox[frame_ind]):
                    if np.array_equal(box, -1 * np.ones(box.shape)):  # only annotated objects
                        continue

                    if box.shape[-1] == 2:  # Operate on point coordinates
                        proc_bbox = np.stack(
                            self.crop_coords(box[:, 0], box[:, 1], self.crop_xmin, self.crop_ymin, self.crop_xmax,
                                             self.crop_ymax), 1)
                    else:  # Operate on bounding box
                        xmin, ymin, xmax, ymax = box
                        proc_bbox = self.crop_bbox(xmin, ymin, xmax, ymax, self.crop_xmin, self.crop_ymin,
                                                   self.crop_xmax, self.crop_ymax)
                    temp_bbox[class_ind, :] = proc_bbox
                out_bbox.append(temp_bbox)

        if bbox:
            return np.array(out_clip), np.array(out_bbox)

        return np.array(out_clip)


class RandomCropClip(PreprocTransform):
    def __init__(self):
        super(RandomCropClip, self).__init__()
        self.crop_h, self.crop_w = config.crop_shape

        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def _update_random_sample(self, frame_h, frame_w):
        if frame_w == self.crop_w:
            self.xmin = 0
        else:
            self.xmin = np.random.randint(0, frame_w - self.crop_w)

        self.xmax = self.xmin + self.crop_w

        if frame_h == self.crop_h:
            self.ymin = 0
        else:
            self.ymin = np.random.randint(0, frame_h - self.crop_h)

        self.ymax = self.ymin + self.crop_h

    def get_random_sample(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    def __call__(self, clip, bbox=None):
        frame_shape = clip[0].shape
        self._update_random_sample(frame_shape[0], frame_shape[1])
        self.crop_transform.update_bbox(self.xmin, self.xmax, self.ymin, self.ymax)
        proc_clip = self.crop_transform(clip, bbox)
        if isinstance(proc_clip, tuple):
            assert (proc_clip[0].shape[1:3] == (self.crop_h, self.crop_w)), 'Proc frame: {} Crop h,w: {},{}'.format(
                proc_clip[0].shape, self.crop_h, self.crop_w)
        else:
            assert (proc_clip.shape[1:3] == (self.crop_h, self.crop_w)), 'Proc frame: {} Crop h,w: {},{}'.format(
                proc_clip.shape, self.crop_h, self.crop_w)
        return proc_clip


class CenterCropClip(PreprocTransform):
    def __init__(self):
        super(CenterCropClip, self).__init__()
        self.crop_h, self.crop_w = config.crop_shape

        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

    def _calculate_center(self, frame_h, frame_w):
        xmin = int(frame_w / 2 - self.crop_w / 2)
        xmax = int(frame_w / 2 + self.crop_w / 2)
        ymin = int(frame_h / 2 - self.crop_h / 2)
        ymax = int(frame_h / 2 + self.crop_h / 2)
        return xmin, xmax, ymin, ymax

    def __call__(self, clip, bbox=None):
        frame_shape = clip[0].shape
        xmin, xmax, ymin, ymax = self._calculate_center(frame_shape[0], frame_shape[1])
        self.crop_transform.update_bbox(xmin, xmax, ymin, ymax)
        proc_clip = self.crop_transform(clip, bbox)
        if isinstance(proc_clip, tuple):
            assert (proc_clip[0].shape[1:3] == (self.crop_h, self.crop_w)), 'Proc frame: {} Crop h,w: {},{}'.format(
                proc_clip[0].shape, self.crop_h, self.crop_w)
        else:
            assert (proc_clip.shape[1:3] == (self.crop_h, self.crop_w)), 'Proc frame: {} Crop h,w: {},{}'.format(
                proc_clip.shape, self.crop_h, self.crop_w)
        return proc_clip


class RandomFlipClip(PreprocTransform):
    """
    Specify a flip direction:
    Horizontal, left right flip = 'h' (Default)
    Vertical, top bottom flip = 'v'
    """

    def __init__(self, direction='h', p=0.5):
        super(RandomFlipClip, self).__init__()
        self.direction = direction
        self.p = p

    def _update_p(self, p):
        self.p = p

    def _random_flip(self):
        flip_prob = np.random.random()
        if flip_prob >= self.p:
            flip_result = 0
        else:
            flip_result = 1

        return flip_result

    def _h_flip(self, bbox, frame_size):
        width = frame_size[1]
        bbox_shape = bbox.shape
        output_bbox = np.zeros(bbox_shape) - 1
        for bbox_ind, box in enumerate(bbox):
            if np.array_equal(box, -1 * np.ones(box.shape)):  # only annotated objects
                continue

            if box.shape[-1] == 2:  # Operate on point coordinates
                x = box[:, 0]
                x_new = width - x

                output_bbox[bbox_ind] = np.stack((x_new, box[:, 1]), 1)
            else:  # Operate on bounding box
                xmin, ymin, xmax, ymax = box
                xmax_new = width - xmin
                xmin_new = width - xmax
                output_bbox[bbox_ind] = xmin_new, ymin, xmax_new, ymax
        return output_bbox

    def _v_flip(self, bbox, frame_size):
        height = frame_size[0]
        bbox_shape = bbox.shape
        output_bbox = np.zeros(bbox_shape) - 1
        for bbox_ind, box in enumerate(bbox):
            if np.array_equal(box, -1 * np.ones(box.shape)):  # only annotated objects
                continue

            if box.shape[-1] == 2:  # Operate on point coordinates
                y = box[:, 1]
                y_new = height - y

                output_bbox[bbox_ind] = np.stack((box[:, 0], y_new), 1)
            else:  # Operate on bounding box
                xmin, ymin, xmax, ymax = box
                ymax_new = height - ymin
                ymin_new = height - ymax
                output_bbox[bbox_ind] = xmin, ymin_new, xmax, ymax_new
        return output_bbox

    def _flip_data(self, clip, bbox):
        output_bbox = []
        output_clip = []
        if self.direction == 'h':
            output_clip = [cv2.flip(np.array(frame), 1) for frame in clip]

            if bbox:
                output_bbox = [self._h_flip(frame, output_clip[0].shape) for frame in bbox]

        elif self.direction == 'v':
            output_clip = [cv2.flip(np.array(frame), 0) for frame in clip]

            if bbox:
                output_bbox = [self._v_flip(frame, output_clip[0].shape) for frame in bbox]

        return output_clip, output_bbox

    def __call__(self, clip, bbox=None):
        input_shape = np.array(clip).shape
        flip = self._random_flip()
        out_clip = clip
        out_bbox = bbox
        if flip:
            out_clip, out_bbox = self._flip_data(clip, bbox)

        out_clip = np.array(out_clip)
        assert (input_shape == out_clip.shape), "Input shape {}, output shape {}".format(input_shape, out_clip.shape)

        # pylint: disable = no-else-return
        if bbox:
            return out_clip, out_bbox
        else:
            return out_clip


class ToTensorClip(PreprocTransform):
    """
    Convert a list of PIL images or numpy arrays to a 5d pytorch tensor [batch, frame, channel, height, width]
    """

    def __call__(self, clip, bbox=None):

        if isinstance(clip[0], Image.Image):
            # a little round-about but it maintains consistency
            temp_clip = []
            for c in clip:
                temp_clip.append(np.array(c))
            clip = temp_clip

        output_clip = mindspore.Tensor.from_numpy(np.array(clip)).float()  # Numpy array to Tensor

        if bbox:
            bbox = mindspore.Tensor.from_numpy(np.array(bbox))
            return output_clip, bbox

        return output_clip


class PreprocessTrainC3D():
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types
            crop_type  (String): Select random or central crop

        Return:
            None
        """

        self.transforms = []
        self.transforms1 = []
        self.preprocess = config.preprocess
        crop_type = config.crop_type

        self.clip_mean = np.load(config.sport1m_mean_file_path)[0]
        self.clip_mean = np.transpose(self.clip_mean, (1, 2, 3, 0))

        self.transforms.append(ResizeClip())
        self.transforms.append(SubtractMeanClip(clip_mean=self.clip_mean))

        if crop_type == 'Random':
            self.transforms.append(RandomCropClip())

        else:
            self.transforms.append(CenterCropClip())

        self.transforms.append(RandomFlipClip(direction='h', p=0.5))

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)

        return input_data


class PreprocessEvalC3D():
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types
            crop_type  (String): Select random or central crop

        Return:
            None
        """

        self.transforms = []
        self.clip_mean = np.load(config.sport1m_mean_file_path)[0]
        self.clip_mean = np.transpose(self.clip_mean, (1, 2, 3, 0))

        self.transforms.append(ResizeClip())
        self.transforms.append(SubtractMeanClip(clip_mean=self.clip_mean))
        self.transforms.append(CenterCropClip())

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)

        return input_data
