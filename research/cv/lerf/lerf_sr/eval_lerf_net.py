# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import dtype as mstype
from PIL import Image

import model

from common.option import TestOptions
from common.resize2d import SGKResize2dNumpy
from common.utils import cal_psnr, cal_ssim, _rgb2ycbcr

mode_pad_dict = {
    "s": 1,
    "d": 2,
    "y": 2,
    "c": 3,
    "t": 3,
}


class Eltr:
    def __init__(self, arg, pred_model):
        self.arg = arg
        self.pred_model = pred_model
        self.norm = 255
        self.out_c = 3  # hyper-param channel
        self.modes = arg.modes
        self.modes2 = arg.modes2
        self.stages = arg.stages
        self.files = None
        self.dataset = None
        self.scale_h = 0
        self.scale_w = 0
        self.result_path = None
        self.resizer = SGKResize2dNumpy(support_sz=arg.suppSize, max_sigma=arg.sigma)

    def run(self, dataset, scale_h, scale_w):
        folder = os.path.join(self.arg.testDir, dataset, "HR")
        files = os.listdir(folder)
        files.sort()

        result_path = os.path.join(
            opt.resultRoot,
            opt.expDir.split("/")[-1],
            "X{:.2f}_{:.2f}".format(scale_h, scale_w),
            dataset,
        )
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.scale_h = scale_h
        self.scale_w = scale_w

        psnr_ssim_list = []
        for i in range(len(self.files)):
            psnr_i, ssim_i = self._worker(i)
            psnr_ssim_list.append([psnr_i, ssim_i])
        return psnr_ssim_list

    def _worker(self, i):
        # Load LR image
        img_lr = np.array(
            Image.open(
                os.path.join(
                    self.arg.testDir,
                    self.dataset,
                    "LR_bicubic/X{}".format(self.scale_h),
                    self.files[i],
                )
            )
        ).astype(np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)

        img_lr = (
            ms.Tensor(img_lr[None, :], dtype=mstype.float32).transpose(0, 3, 1, 2)
            / 255.0
        )
        # Load GT image
        img_gt = np.array(
            Image.open(os.path.join(self.arg.testDir, self.dataset, "HR", self.files[i]))
        )

        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)

        # split channels to avoid fold operator
        img_pred = []
        for j in range(3):
            img_lr_c = self.pred_model.predict(img_lr[:, j : j + 1, :, :], stage=1)
            img_hyper = self.pred_model.predict(img_lr_c / 255.0, stage=2).numpy()[0]
            img_lr_c = img_lr_c[0].numpy()
            self.resizer.set_shape(
                img_lr_c.shape, scale_factors=[self.scale_h, self.scale_w]
            )
            img_pred.append(
                self.resizer.resize(
                    img_lr_c, img_hyper[:1], img_hyper[1:2], img_hyper[2:]
                )
            )

        img_out = np.concatenate(img_pred, axis=0)

        img_out = np.clip(np.round(img_out).transpose((1, 2, 0)), 0, self.norm).astype(
            np.uint8
        )

        # Save to file
        Image.fromarray(img_out).save(
            os.path.join(
                self.result_path, "{}_net.png".format(self.files[i].split("/")[-1][:-4])
            )
        )

        if img_gt.shape != img_out.shape:
            pred_h, pred_w, _ = img_out.shape
            img_gt = img_gt[:pred_h, :pred_w, :]
        y_gt, y_out = _rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0]
        psnr = cal_psnr(y_gt, y_out, max(int(self.scale_h), int(self.scale_w)))
        ssim = cal_ssim(y_gt, y_out)
        return [psnr, ssim]


if __name__ == "__main__":
    opt = TestOptions().parse()

    context.set_context(mode=context.GRAPH_MODE, device_id=0, save_graphs=False)
    context.set_context(device_target="GPU")

    modes = [item for item in opt.modes]
    stages = opt.stages

    model = getattr(model, opt.model)

    model_g = model(opt, out_c=3)

    # Load saved network params
    ckpt_file = os.path.join(opt.expDir, "Model_{:06d}.ckpt".format(opt.loadIter))

    if os.path.exists(ckpt_file):
        all_param = ms.load_checkpoint(ckpt_file)
        for s in range(opt.stages):
            if (s + 1) == opt.stages:
                for mode in opt.modes2:
                    for r in [0, 1]:
                        name = "s{}_{}r{}".format(str(s + 1), mode, r)
                        param_dict = {
                            k.replace(name + ".", ""): v
                            for k, v in all_param.items()
                            if name in k
                        }
                        param_not_load = ms.load_param_into_net(
                            getattr(model_g, name), param_dict
                        )
            else:
                for mode in opt.modes:
                    name = "s{}_{}r0".format(str(s + 1), mode)
                    param_dict = {
                        k.replace(name + ".", ""): v
                        for k, v in all_param.items()
                        if name in k
                    }
                    param_not_load = ms.load_param_into_net(
                        getattr(model_g, name), param_dict
                    )

        etr = Eltr(opt, model_g)

        all_datasets = ["Set5"]
        all_scales = [[2, 2], [3, 3], [4, 4]]

        scale_head = ["Scale".ljust(15, " ")]
        for scale_p in all_scales:
            scale_h_j, scale_w_j = scale_p
            scale_head.append("{:.1f}x{:.1f}\t".format(scale_h_j, scale_w_j))
        print("\t".join(scale_head))

        for dataset_i in all_datasets:
            metric_list = [dataset_i.ljust(15, " ")]
            for scale_p in all_scales:
                scale_h_i, scale_w_i = scale_p
                psnr_ssim_s = etr.run(dataset_i, scale_h_i, scale_w_i)
                avg_psnr, avg_ssim = np.mean(np.asarray(psnr_ssim_s)[:, 0]), np.mean(
                    np.asarray(psnr_ssim_s)[:, 1]
                )
                metric_list.append("{:.2f}/{:.4f}".format(avg_psnr, avg_ssim))
            print("\t".join(metric_list))
    else:
        print("Please download the model checkpoint first.")
