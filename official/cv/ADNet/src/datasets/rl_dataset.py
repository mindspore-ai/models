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
# pytorch dataset for SL learning
# matlab code (line 26-33):
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference:
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py

import time
import cv2
import numpy as np

from src.trainers.RL_tools import TrackingEnvironment
from src.utils.augmentations import ADNet_Augmentation
from src.utils.display import display_result, draw_box

from mindspore import Tensor, ops
from mindspore import dtype as mstype


class RLDataset:

    def __init__(self, net, domain_specific_nets, train_videos, opts, args):
        self.env = None

        # these lists won't include the ground truth
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []
        self.vid_idx_list = []

        self.reset(net, domain_specific_nets, train_videos, opts, args)

    def __getitem__(self, index):
        index = index % len(self.log_probs_list)
        return np.array(self.log_probs_list[index]), \
               np.array(self.reward_list[index]), \
               np.array(self.vid_idx_list[index]), \
               np.array(self.patch_list[index])

    def __len__(self):
        return len(self.log_probs_list)

    def reset(self, net, domain_specific_nets, train_videos, opts, args):
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []
        self.vid_idx_list = []

        print('generating reinforcement learning dataset')
        transform = ADNet_Augmentation(opts)

        self.env = TrackingEnvironment(train_videos, opts, transform=transform, args=args)
        clip_idx = 0
        while True:  # for every clip (l)

            num_step_history = []  # T_l

            num_frame = 1  # the first frame won't be tracked..
            t = 0
            box_history_clip = []  # for checking oscillation in a clip
            net.reset_action_dynamic()  # action dynamic should be in a clip (what makes sense...)

            while True:  # for every frame in a clip (t)
                tic = time.time()

                if args.display_images:
                    im_with_bb = display_result(self.env.get_current_img(), self.env.get_state())
                    cv2.imshow('patch', self.env.get_current_patch_unprocessed())
                    cv2.waitKey(1)
                else:
                    im_with_bb = draw_box(self.env.get_current_img(), self.env.get_state())

                if args.save_result_images:
                    cv2.imwrite('images/' + str(clip_idx) + '-' + str(t) + '.jpg', im_with_bb)

                curr_patch = self.env.get_current_patch()
                self.patch_list.append(curr_patch)
                curr_patch = Tensor(np.expand_dims(curr_patch, 0), mstype.float32).transpose(0, 3, 1, 2)

                # load ADNetDomainSpecific with video index
                if args.multidomain:
                    vid_idx = self.env.get_current_train_vid_idx()
                else:
                    vid_idx = 0
                net.load_domain_specific(domain_specific_nets[vid_idx])

                fc6_out, _ = net(curr_patch, -1, True)
                net.update_action_dynamic(net.action_history)

                action = np.argmax(fc6_out.asnumpy())
                log_prob = ops.Log()(Tensor(fc6_out[0][Tensor(action, mstype.int32)].asnumpy(), mstype.float32))

                self.log_probs_list.append(np.asscalar(log_prob.asnumpy()))
                if args.multidomain:
                    self.vid_idx_list.append(np.asscalar(vid_idx))
                else:
                    self.vid_idx_list.append(0)

                self.action_list.append(action)

                new_state, reward, done, info = self.env.step(action)
                if done and info['finish_epoch']:
                    pass
                # check oscillating
                elif any((np.array(new_state).round() == x).all() for x in np.array(box_history_clip).round()):
                    action = opts['stop_action']
                    reward, done, finish_epoch = self.env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                # check if number of action is already too much
                if t > opts['num_action_step_max']:
                    action = opts['stop_action']
                    reward, done, finish_epoch = self.env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                box_history_clip.append(list(new_state))

                t += 1

                if action == opts['stop_action']:
                    num_frame += 1
                    num_step_history.append(t)
                    t = 0

                toc = time.time() - tic
                print('forward time (clip ' + str(clip_idx) + " - frame " + str(num_frame) + " - t " + str(t) + ") = "
                      + str(toc) + " s")

                if done:  # if finish the clip
                    break

            tracking_scores_size = np.array(num_step_history).sum()
            tracking_scores = np.full(tracking_scores_size, reward)  # seems no discount factor whatsoever

            self.reward_list.extend(tracking_scores)

            clip_idx += 1

            if info['finish_epoch']:
                break

        print('generating reinforcement learning dataset finish')
