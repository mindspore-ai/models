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
"""net train"""

import datetime
import os
import random

import numpy as np
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import save_checkpoint
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

import src.dataset as dt
from src.config import relationnet_cfg as cfg

scatter = ops.ScatterNd()
concat0dim = ops.Concat(axis=0)


def train(metatrain_character_folders, metatest_character_folders, netloss, net_g, encoder_relation,
          local_train_url, args):
    '''train model'''
    print("=" * 10 + "Training" + "=" * 10)
    last_accuracy = 0.0
    last_accuracy_episode = 0
    episodes_1k_running_times = []
    episode_5k_start_time = datetime.datetime.now()
    for episode in range(cfg.episode):
        episode_start_time = datetime.datetime.now()

        degrees = random.choice([0, 90, 180, 270])
        flip = random.choice([True, False])
        task = dt.OmniglotTask(metatrain_character_folders, cfg.class_num, cfg.sample_num_per_class,
                               cfg.batch_num_per_class)
        sample_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="train",
                                               shuffle=False, rotation=degrees, flip=flip)
        batch_daraloader = dt.get_data_loader(task, num_per_class=cfg.batch_num_per_class, split='test',
                                              shuffle=True, rotation=degrees, flip=flip)

        samples, _ = sample_dataloader.__iter__().__next__()
        batches, batches_labels = batch_daraloader.__iter__().__next__()
        # concat samples and batches

        train_input = concat0dim((samples, batches))

        # precess y_hot
        update = Tensor(np.array([1 for _ in range(95)]), mstype.float32)
        shape = (cfg.batch_num_per_class * cfg.class_num, cfg.class_num)
        indices = Tensor(np.array([[i, j] for i, j in zip([i for i in range(batches_labels.shape[0])],
                                                          [j for j in batches_labels.asnumpy()])]), mstype.int32)
        one_hot_labels = scatter(indices, update, shape)

        # calculate loss and backward
        loss = netloss(train_input, one_hot_labels)
        _ = net_g(train_input, one_hot_labels)

        episode_end_time = datetime.datetime.now()
        episodes_1k_running_times.append((episode_end_time - episode_start_time).total_seconds())

        if (episode + 1) % 1000 == 0:
            print('-' * 5 + 'Episode {}/{}'.format(episode + 1, cfg.episode) + '-' * 5)
            print('Episode: {} Train, Loss(MSE): {}, One episode average time: {} msec, 1000 episodes time: {} sec'
                  .format(
                      episode + 1,
                      loss,
                      np.mean(episodes_1k_running_times) * 1000,
                      np.sum(episodes_1k_running_times)
                  ))
            episodes_1k_running_times = []

        if (episode + 1) % 5000 == 0:
            testing_start_time = datetime.datetime.now()
            episode_5k_end_time = datetime.datetime.now()
            print(f"5000 episodes time: {(episode_5k_end_time - episode_5k_start_time).total_seconds()} sec")
            print("=" * 10 + "Testing" + "=" * 10)
            total_rewards = 0

            for _ in range(cfg.test_episode):
                degrees = random.choice([0, 90, 180, 270])
                flip = random.choice([True, False])
                task = dt.OmniglotTask(metatest_character_folders, cfg.class_num, cfg.sample_num_per_class,
                                       cfg.sample_num_per_class)
                sample_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="train",
                                                       shuffle=False, rotation=degrees, flip=flip)
                test_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="test",
                                                     shuffle=True, rotation=degrees, flip=flip)
                test_samples, _ = sample_dataloader.__iter__().__next__()
                test_batches, test_batch_labels = test_dataloader.__iter__().__next__()
                encoder_relation.set_train(False)
                # concat samples and batches
                test_input = concat0dim((test_samples, test_batches))
                test_relations = encoder_relation(test_input)

                predict_labels = P.Argmax(axis=1, output_type=mstype.int32)(test_relations).asnumpy()
                test_batch_labels = test_batch_labels.asnumpy().astype(np.int32)
                rewards = [1 if predict_labels[j] == test_batch_labels[j] else 0 for j in range(cfg.class_num)]
                total_rewards += np.sum(rewards)

            encoder_relation.set_train(True)
            test_accuracy = total_rewards / 1.0 / cfg.class_num / cfg.test_episode

            print("test accuracy: %.4f or %.4f%%" % (test_accuracy, test_accuracy * 100))

            if test_accuracy > last_accuracy:

                # save networks
                save_checkpoint(encoder_relation, os.path.join(local_train_url,
                                                               str("omniglot_encoder_relation_network" +
                                                                   str(cfg.class_num) + "way_" +
                                                                   str(cfg.sample_num_per_class) + "shot.ckpt")))
                if args.cloud:
                    import moxing as mox
                    mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
                print("save networks for episode : {}".format(episode + 1))
                last_accuracy = test_accuracy
                last_accuracy_episode = episode + 1
            print("Best_accuracy : %.4f  in Episode : %d" % (last_accuracy, last_accuracy_episode))

            testing_end_time = datetime.datetime.now()
            print(f"Testing time: {(testing_end_time - testing_start_time).total_seconds()} sec")
            episode_5k_start_time = datetime.datetime.now()
