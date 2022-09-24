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

from src.loss import BoxLoss, LandmarkLoss, ClassLoss
import config as cfg

from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeNormal, Uniform


class PNet(nn.Cell):
    """fast Proposal Network(P-Net)"""
    def __init__(self):
        super(PNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')

        self.conv2 = nn.Conv2d(10, 16, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(16, 32, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu3 = nn.PReLU()

        # detection
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1, has_bias=True, pad_mode='valid')
        # bounding box regression
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1, has_bias=True, pad_mode='valid')
        # landmark regression
        self.conv4_3 = nn.Conv2d(32, 10, 1, 1, has_bias=True, pad_mode='valid')

        for cell in self.cells():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'), cell.weight.shape))
                cell.bias.set_data(initializer(Uniform(), cell.bias.shape))
        self.squeeze = ops.Squeeze()

    def construct(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))

        # Output classification result
        label = self.conv4_1(x)
        # box regression
        offset = self.conv4_2(x)
        # landmark regression
        landmark = self.conv4_3(x)
        return self.squeeze(label), self.squeeze(offset), self.squeeze(landmark)

class PNetWithLoss(nn.Cell):
    """PNet with loss cell"""
    def __init__(self):
        super(PNetWithLoss, self).__init__()
        self.net = PNet()
        self.cls_loss = ClassLoss()
        self.box_loss = BoxLoss()
        self.landmark_loss = LandmarkLoss()

    def construct(self, x, gt_label, gt_box, gt_landmark):
        pred_label, pred_box, pred_landmark = self.net(x)
        assert pred_label.ndim == 2 and pred_box.ndim == 2 and pred_landmark.ndim == 2, "Need to squeeze"
        cls_loss_value = self.cls_loss(gt_label, pred_label)
        box_loss_value = self.box_loss(gt_label, gt_box, pred_box)
        landmark_loss_value = self.landmark_loss(gt_label, gt_landmark, pred_landmark)
        total_loss = cfg.RADIO_CLS_LOSS * cls_loss_value + \
                     cfg.RADIO_BOX_LOSS * box_loss_value + cfg.RADIO_LANDMARK_LOSS * landmark_loss_value
        return total_loss


class PNetTrainOneStepCell(nn.Cell):
    """PNet Train One Step Cell"""
    def __init__(self, network, optimizer):
        super(PNetTrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        self.optimizer(grads)
        return loss

class RNet(nn.Cell):
    """Refinement Network(R-Net)"""
    def __init__(self):
        super(RNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 28, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2, pad_mode='same')

        self.conv2 = nn.Conv2d(28, 48, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2, pad_mode='valid')

        self.conv3 = nn.Conv2d(48, 64, 2, 1, has_bias=True, pad_mode='valid')
        self.prelu3 = nn.PReLU()
        self.flatten = nn.Flatten()

        self.fc = nn.Dense(576, 128)

        # detection
        self.class_fc = nn.Dense(128, 2)
        # bounding box regression
        self.bbox_fc = nn.Dense(128, 4)
        # landmark localization
        self.landmark_fc = nn.Dense(128, 10)

        for cell in self.cells():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'), cell.weight.shape))
                cell.bias.set_data(initializer(Uniform(), cell.bias.shape))

    def construct(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))

        x = self.flatten(x)
        x = self.fc(x)

        # detection
        det = self.class_fc(x)
        box = self.bbox_fc(x)
        landmark = self.landmark_fc(x)

        return det, box, landmark

class RNetWithLoss(nn.Cell):
    def __init__(self):
        super(RNetWithLoss, self).__init__()
        self.net = RNet()
        self.cls_loss = ClassLoss()
        self.box_loss = BoxLoss()
        self.landmark_loss = LandmarkLoss()

    def construct(self, x, gt_label, gt_box, gt_landmark):
        pred_label, pred_box, pred_landmark = self.net(x)
        cls_loss_value = self.cls_loss(gt_label, pred_label)
        box_loss_value = self.box_loss(gt_label, gt_box, pred_box)
        landmark_loss_value = self.landmark_loss(gt_label, gt_landmark, pred_landmark)
        total_loss = cfg.RADIO_CLS_LOSS * cls_loss_value + \
                     cfg.RADIO_BOX_LOSS * box_loss_value + cfg.RADIO_LANDMARK_LOSS * landmark_loss_value
        return total_loss

class RNetTrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(RNetTrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        self.optimizer(grads)
        return loss

class ONet(nn.Cell):
    """Output Network(O-Net)"""
    def __init__(self):
        super(ONet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2, pad_mode='same')

        self.conv2 = nn.Conv2d(32, 64, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2, pad_mode='valid')

        self.conv3 = nn.Conv2d(64, 64, 3, 1, has_bias=True, pad_mode='valid')
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, 2, pad_mode='valid')

        self.conv4 = nn.Conv2d(64, 128, 2, 1, has_bias=True, pad_mode='valid')
        self.prelu4 = nn.PReLU()

        self.fc = nn.Dense(1152, 256)

        self.flatten = nn.Flatten()

        # detection
        self.class_fc = nn.Dense(256, 2)
        # bounding box regression
        self.bbox_fc = nn.Dense(256, 4)
        # landmark localization
        self.landmark_fc = nn.Dense(256, 10)

        for cell in self.cells():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'), cell.weight.shape))
                cell.bias.set_data(initializer(Uniform(), cell.bias.shape))

    def construct(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))
        x = self.flatten(x)
        x = self.fc(x)
        # detection
        det = self.class_fc(x)
        # box regression
        box = self.bbox_fc(x)
        # landmark regression
        landmark = self.landmark_fc(x)

        return det, box, landmark

class ONetWithLoss(nn.Cell):
    def __init__(self):
        super(ONetWithLoss, self).__init__()
        self.net = ONet()
        self.cls_loss = ClassLoss()
        self.box_loss = BoxLoss()
        self.landmark_loss = LandmarkLoss()

    def construct(self, x, gt_label, gt_box, gt_landmark):
        pred_label, pred_box, pred_landmark = self.net(x)
        cls_loss_value = self.cls_loss(gt_label, pred_label)
        box_loss_value = self.box_loss(gt_label, gt_box, pred_box)
        landmark_loss_value = self.landmark_loss(gt_label, gt_landmark, pred_landmark)
        total_loss = 1.0 * cls_loss_value + 0.5 * box_loss_value + 1.0 * landmark_loss_value
        return total_loss

class ONetTrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(ONetTrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        self.optimizer(grads)
        return loss
