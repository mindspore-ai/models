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
"""network"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from src.head import ClsCntRegHead
from src.fpn_neck import FPN
from src.resnet import resnet50
from src.config import DefaultConfig
from src.network_define import coords_fmap2orig



class FCOS(nn.Cell):
    def __init__(self, config=None, preckpt_path=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet50(pretrained=config.pretrained, preckpt_path=preckpt_path)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels, config.class_num,
                                  config.use_GN_head, config.cnt_on_reg, config.prior)
        self.config = config

        self.freeze()

    def train(self, mode=True):
        """
        set module training mode, and frozen bn
        """
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)

    def flatten(self, nested):
        try:
            try:
                nested + ''
            except TypeError:
                pass
            else:
                raise TypeError

            for sublist in nested:
                for element in self.flatten(sublist):
                    yield element
        except TypeError:
            yield nested

    def freeze(self):

        for i in self.trainable_params():
            if i.name.find('bn') != -1 or i.name.find('down_sample_layer.1') != -1:
                i.requires_grad = False

        self.backbone.freeze_stages(1)



    def construct(self, x):
        """
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn((C3, C4, C5))
        cls_logits, cnt_logits, reg_preds = self.head((all_P))
        return (cls_logits, cnt_logits, reg_preds)


class DetectHead(nn.Cell):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def construct(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''

        cast = ops.Cast()
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]\

        sigmoid = ops.Sigmoid()

        cls_preds = sigmoid(cls_logits)
        cnt_preds = sigmoid(cnt_logits)

        cls_classes, cls_scores = ops.ArgMaxWithValue(axis=-1)(cls_preds)  # [batch_size,sum(_h*_w)]

        cnt_preds = ops.Squeeze(axis=-1)(cnt_preds)
        cls_scores = ops.Sqrt()(cls_scores * cnt_preds)
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]
        if self.max_detection_boxes_num > cls_scores.shape[-1]:
            max_num = cls_scores.shape[-1]
        else:
            max_num = self.max_detection_boxes_num
        topk = ops.TopK(sorted=True)
        topk_ind = topk(cls_scores, max_num)[1]  # [batch_size,max_num]

        _cls_scores = ()
        _cls_classes = ()
        _boxes = ()
        stack = mindspore.ops.Stack(axis=0)
        for batch in range(cls_scores.shape[0]):
            topk_index = cast(topk_ind, mindspore.int32)
            _cls_scores = _cls_scores + (cls_scores[batch][topk_index],)  # [max_num]
            _cls_classes = _cls_classes + (cls_classes[batch][topk_index],)  # [max_num]
            _boxes = _boxes + (boxes[batch][topk_index],)  # [max_num,4]
        cls_scores_topk = stack(_cls_scores)#[batch_size,max_num]
        cls_classes_topk = stack(_cls_classes)#[batch_size,max_num]
        boxes_topk = stack(_boxes)#[batch_size,max_num,4]
        return cls_scores_topk, cls_classes_topk, boxes_topk


    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        concat = ops.Concat(axis=-1)
        boxes = concat((x1y1, x2y2))  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = ()
        coords = ()
        reshape = ops.Reshape()
        transpose = ops.Transpose()
        for pred, stride in zip(inputs, strides):
            input_perm = (0, 2, 3, 1)
            pred = transpose(pred, input_perm)
            coord = coords_fmap2orig(pred, stride)
            pred = reshape(pred, (batch_size, -1, c))
            out = out + (pred,)
            coords = coords + (coord,)
        return ops.Concat(axis=1)(out), ops.Concat(axis=0)(coords)


class FCOSDetector(nn.Cell):
    def __init__(self, mode, config=None, preckpt_path=None):
        super().__init__()
        config = DefaultConfig
        self.mode = mode
        self.fcos_body = FCOS(config=config, preckpt_path=preckpt_path)
        if mode == "training":
            pass
        elif mode == "inference":
            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold, \
            config.max_detection_boxes_num, config.strides, config)

    def construct(self, input_imgs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''
        out = self.fcos_body(input_imgs)
        if self.mode != "training":
            scores, classes, boxes = self.detection_head(out)
            out = (scores, classes, boxes)
        return out
