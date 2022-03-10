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
"""callbacks"""
from sklearn.metrics import roc_auc_score
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from src.utils import cal_anomaly_map


class EvalCallBack(Callback):
    """EvalCallBack"""

    def __init__(self, dataset, net, args, save_path=None):
        self.dataset = dataset
        self.network = net
        self.network.set_train(False)
        self.start_epoch = args.start_eval_epoch
        self.save_path = save_path
        self.interval = args.eval_interval
        self.best_pixel = 0
        self.best_image = 0
        self.category = args.category
        self.best_epoch = 0
        self.best_image_added = 0
        self.best_pixel_added = 0
        self.best_epoch_added = 0
        self.end_epoch = args.epoch
        self.out_size = args.out_size

    def epoch_end(self, run_context):
        """epoch_end"""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        file_name = "../final.log"
        if cur_epoch >= self.start_epoch:
            if (cur_epoch - self.start_epoch) % self.interval == 0:
                self.network.set_train(False)
                pixel_auc, img_auc = self.inference()
                if (pixel_auc + img_auc) >= (self.best_pixel_added + self.best_image_added):
                    self.best_image_added = img_auc
                    self.best_pixel_added = pixel_auc
                    self.best_epoch_added = cur_epoch
                    save_checkpoint(self.network, './ckpt/' +
                                    self.category + "_best_added.ckpt")
                print("\033[1;34m", flush=True)
                print("epoch: {},  pixel auc: {}, image auc: {}. "
                      "best pixel auc: {}, best image auc: {} "
                      "at {} epoch".format(cur_epoch, pixel_auc, img_auc, self.best_pixel_added,
                                           self.best_image_added, self.best_epoch_added), flush=True)
                print("\033[0m", flush=True)
                if cur_epoch == self.end_epoch:
                    with open(file_name, "a+") as f:
                        f.write("category: {}, best pixel auc: {}, best image auc: {} \r\n".format(
                            self.category, self.best_pixel_added, self.best_image_added))
                    print("\033[1;31m", flush=True)
                    print("epoch: {},  pixel auc: {}, image auc: {}. "
                          "best pixel auc: {}, best image auc: {} at {} epoch. "
                          "best pixel added auc: {}, best image added auc: {} "
                          "at {} epoch".format(cur_epoch, pixel_auc, img_auc,
                                               self.best_pixel, self.best_image, self.best_epoch,
                                               self.best_pixel_added, self.best_image_added,
                                               self.best_epoch_added), flush=True)
                    print("\033[0m", flush=True)

    def inference(self):
        "inference"
        gt_list_px_lvl = []
        pred_list_px_lvl = []
        gt_list_img_lvl = []
        pred_list_img_lvl = []
        for data in self.dataset.create_dict_iterator():
            gt = data['gt']
            label = data['label']
            features_s, features_t = self.network(data['img'])
            anomaly_map = cal_anomaly_map(features_s, features_t, out_size=self.out_size)
            gt_np = gt.asnumpy()[0, 0].astype(int)

            gt_list_px_lvl.extend(gt_np.ravel())
            pred_list_px_lvl.extend(anomaly_map.ravel())
            gt_list_img_lvl.append(label.asnumpy()[0])
            pred_list_img_lvl.append(anomaly_map.max())
        pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
        img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)
        return pixel_auc, img_auc
