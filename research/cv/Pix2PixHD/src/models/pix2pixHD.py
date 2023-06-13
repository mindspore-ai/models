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
# ===========================================================================

"""
    Define Pix2PixHD model.
"""
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from src.utils.config import config
from src.utils.tools import load_network
from .network import get_generator, get_discriminator
from .loss import GANLoss, VGGLoss


class Pix2PixHD(nn.Cell):
    def __init__(self, is_train=True):
        super(Pix2PixHD, self).__init__(auto_prefix=True)
        self.concat = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.one_hot = ops.OneHot(axis=-1)
        self.cast = ops.Cast()
        self.is_train = is_train
        self.use_features = config.instance_feat or config.label_feat
        self.gen_features = self.use_features and not config.load_features
        self.no_vgg_loss = config.no_vgg_loss
        self.no_ganFeat_loss = config.no_ganFeat_loss
        self.load_features = config.load_features
        self.n_layers_D = config.n_layers_D
        self.num_D = config.num_D
        self.lambda_feat = config.lambda_feat
        self.use_encoded_image = config.use_encoded_image
        self.save_ckpt_dir = config.save_ckpt_dir
        self.name = config.name
        self.cluster_path = config.cluster_path
        self.feat_num = config.feat_num
        self.label_nc = config.label_nc
        self.input_nc = config.label_nc if config.label_nc != 0 else config.input_nc
        self.initialize()

    def initialize(self):
        # Generator network
        netG_input_nc = self.input_nc
        if not config.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += config.feat_num

        self.netG = get_generator(
            netG_input_nc,
            config.output_nc,
            config.ngf,
            config.netG,
            config.n_downsample_global,
            config.n_blocks_global,
            config.n_local_enhancers,
            config.n_blocks_local,
            config.norm,
        )

        # Encoder network
        if self.gen_features:
            self.netE = get_generator(
                config.output_nc, config.feat_num, config.nef, "encoder", config.n_downsample_E, norm=config.norm
            )

        if self.is_train:
            # Discriminator network
            netD_input_nc = self.input_nc + config.output_nc
            if not config.no_instance:
                netD_input_nc += 1
            self.netD = get_discriminator(
                netD_input_nc,
                config.ndf,
                config.n_layers_D,
                config.norm,
                config.no_lsgan,
                config.num_D,
                not config.no_ganFeat_loss,
            )
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not config.no_lsgan)
            self.criterionFeat = nn.L1Loss()
            if not config.no_vgg_loss:
                self.criterionVGG = VGGLoss()
            # optimizer G
            if config.niter_fix_global > 0:
                params_dict = self.netG.parameters_dict()
                for key, value in params_dict.items():
                    if not key.startswith("netG.model_local"):
                        value.requires_grad = False
            params = list(self.netG.trainable_params())
            if self.gen_features:
                params += list(self.netE.trainable_params())
            self.trainable_params_G = params
            self.trainable_params_D = self.netD.trainable_params()

        # load networks
        if not self.is_train or config.continue_train or config.load_pretrain:
            pretrain_path = "" if not self.is_train else config.load_pretrain
            load_network(self.netG, "G", config.which_epoch, pretrain_path)
            if self.is_train:
                load_network(self.netD, "D", config.which_epoch, pretrain_path)
            if self.gen_features:
                load_network(self.netE, "E", config.which_epoch, pretrain_path)

    def construct(self, input_label, inst_map, real_image, feat_map):
        # Fake Generation
        if self.use_features:
            if self.is_train:
                if not self.load_features:
                    feat_map = self.netE(real_image, inst_map)
                input_concat = self.concat((input_label, feat_map))
            else:
                if self.use_encoded_image:
                    # encode the real image to get feature map
                    feat_map = self.netE(real_image, inst_map)
                input_concat = self.concat((input_label, feat_map))
        else:
            input_concat = input_label
        fake_image = self.netG(input_concat)
        return fake_image

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None):
        label_map = ms.Tensor(label_map)

        if config.label_nc == 0:
            input_label = label_map
        else:
            # create one-hot vector for label map
            input_label = self.one_hot_encode(label_map)

        # get edges from instance map
        if not config.no_instance:
            edge_map = self.get_edges(inst_map)
            input_label = self.concat((input_label, edge_map))

        if inst_map is not None:
            inst_map = ms.Tensor(inst_map)

        if real_image is not None:
            real_image = ms.Tensor(real_image)

        if feat_map is not None:
            feat_map = ms.Tensor(feat_map)

        return input_label, inst_map, real_image, feat_map

    def get_edges(self, t):
        edge = self.zeros(t.shape, ms.int32).asnumpy()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return ms.Tensor(edge, dtype=ms.float32)

    @ms.jit()
    def one_hot_encode(self, label_map):
        size = label_map.shape
        label_map = self.cast(label_map, ms.int32)
        label_map = label_map.reshape(size[0], size[2], size[3])
        depth, on_value, off_value = self.label_nc, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
        one_hot = self.one_hot(label_map, depth, on_value, off_value)
        one_hot = one_hot.transpose(0, 3, 1, 2)
        return one_hot

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.save_ckpt_dir, self.name, self.cluster_path)
        features_clustered = np.load(cluster_path, encoding="latin1", allow_pickle=True).item()

        # randomly sample from the feature clusters
        inst_np = inst.asnumpy().astype(int)
        feat_map = ms.Tensor(np.random.randn(inst.shape[0], self.feat_num, inst.shape[2], inst.shape[3]), ms.float32)
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = ms.Tensor(np.transpose((inst_np == i).nonzero()), ms.int32)
                for k in range(self.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        return feat_map

    def encode_features(self, image, inst):
        image = ms.Tensor(image)
        feat_num = self.feat_num
        h, w = inst.shape[2], inst.shape[3]
        block_num = 32
        feat_map = self.netE(image, ms.Tensor(inst))
        inst_np = inst.astype(int)
        feature = {}
        for i in range(config.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = ms.Tensor(np.transpose((inst == i).nonzero()), ms.int32)
            num = idx.shape[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].asnumpy()
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def reset_netG_grads(self, require_grad):
        """reset_grad"""
        for param in self.netG.get_parameters():
            param.requires_grad = require_grad
        params = list(self.netG.trainable_params())
        if self.gen_features:
            params += list(self.netE.trainable_params())
        self.trainable_params_G = params
