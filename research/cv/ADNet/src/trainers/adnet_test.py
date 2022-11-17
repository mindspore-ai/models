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
# ADNet/adnet_test.m
import os
from random import shuffle
import time

import glob
import cv2
import numpy as np

from src.models.ADNet import WithLossCell_ADNET
from src.datasets.online_adaptation_dataset import OnlineAdaptationDataset, OnlineAdaptationDatasetStorage
from src.utils.augmentations import ADNet_Augmentation
from src.utils.do_action import do_action
from src.utils.display import display_result, draw_box
from src.utils.gen_samples import gen_samples
from src.utils.get_wrapper_utils import get_groundtruth, get_dataLoader

from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.nn import TrainOneStepCell
from mindspore.communication.management import get_rank, get_group_size


def adnet_test(net, vid_path, opts, args):
    print('Testing sequences in ' + str(vid_path) + '...')
    t_sum = 0
    vid_info = {'gt': [], 'img_files': glob.glob(os.path.join(vid_path, 'img', '*.jpg')), 'nframes': 0}
    vid_info['img_files'].sort(key=str.lower)
    gt_path = os.path.join(vid_path, 'groundtruth_rect.txt')
    gt = get_groundtruth(gt_path)
    vid_info['gt'] = gt
    if vid_info['gt'][-1] == '':  # small hack
        vid_info['gt'] = vid_info['gt'][:-1]
    vid_info['nframes'] = min(len(vid_info['img_files']), len(vid_info['gt']))
    # catch the first box
    curr_bbox = vid_info['gt'][0]
    bboxes = np.zeros(np.array(vid_info['gt']).shape)  # tracking result containers init containers
    ntraining = 0
    # setup training
    net_action, net_score = get_optim(net, ops.SparseSoftmaxCrossEntropyWithLogits(), opts)

    dataset_storage_pos = None
    dataset_storage_neg = None
    is_negative = False  # is_negative = True if the tracking failed
    target_score = 0
    all_iteration = 0
    t = 0

    for idx in range(vid_info['nframes']):
        frame_idx = idx
        frame_path = vid_info['img_files'][idx]
        t0_wholetracking = time.time()
        frame = cv2.imread(frame_path)
        # draw box or with display, then save
        im_with_bb = display_result(frame, curr_bbox) if args.display_images else draw_box(frame, curr_bbox)
        save_img(args, os.path.join(args.save_result_images, str(frame_idx) + '-' + str(t) + '.jpg'), im_with_bb)
        curr_bbox_old = curr_bbox
        cont_negatives = 0

        if frame_idx > 0:
            curr_score, cont_negatives = detection(net, opts, args, frame, curr_bbox,
                                                   cont_negatives, frame_idx, ntraining)
            print('final curr_score: %.4f' % curr_score)

            # redetection when confidence < threshold 0.5. But when fc7 is already reliable. Else, just trust the ADNet
            if ntraining > args.believe_score_result:
                if curr_score < 0.5:
                    is_negative = True
                    redetection(net, curr_bbox_old, opts, cont_negatives, frame, args, frame_idx)
                else:
                    is_negative = False
            else:
                is_negative = False
        save_img(args, os.path.join(args.save_result_images, 'final-' + str(frame_idx) + '.jpg'), im_with_bb)

        # record the curr_bbox result
        bboxes[frame_idx] = curr_bbox

        # create or update storage + set iteration_range for training
        if frame_idx == 0:
            dataset_storage_pos = OnlineAdaptationDatasetStorage(
                initial_frame=frame, first_box=curr_bbox, opts=opts, args=args, positive=True)
            # (thanks to small hack in adnet_test) the nNeg_online is also 0
            dataset_storage_neg = OnlineAdaptationDatasetStorage(
                initial_frame=frame, first_box=curr_bbox, opts=opts, args=args, positive=False)

            iteration_range = range(opts['finetune_iters'])
        else:
            assert dataset_storage_pos is not None
            # (thanks to small hack in adnet_test) the nNeg_online is also 0
            assert dataset_storage_neg is not None
            # if confident or when always generate samples, generate new samples
            always_generate_samples = (ntraining < args.believe_score_result)

            if always_generate_samples or (not is_negative or target_score > opts['successThre']):
                dataset_storage_pos.add_frame_then_generate_samples(frame, curr_bbox)

            iteration_range = range(opts['finetune_iters_online'])

        # training when depend on the frequency.. else, don't run the training code...
        if frame_idx % args.online_adaptation_every_I_frames == 0:
            ntraining += 1
            # generate dataset just before training
            dataset_pos = OnlineAdaptationDataset(dataset_storage_pos)
            dataloader_pos = get_dataLoader(dataset_pos, opts, args, ["im", "bbox", "action_label", "score_label"])
            batch_iterator_pos = iter(dataloader_pos)

            # (thanks to small hack in adnet_test) the nNeg_online is also 0
            dataset_neg = OnlineAdaptationDataset(dataset_storage_neg)
            dataloader_neg = get_dataLoader(dataset_neg, opts, args, ["im", "bbox", "action_label", "score_label"])
            batch_iterator_neg = iter(dataloader_neg)
            # else:
            #     dataset_neg = []

            epoch_size_pos = len(dataset_pos) // opts['minibatch_size']
            epoch_size_neg = len(dataset_neg) // opts['minibatch_size']
            if args.distributed:
                rank_id = get_rank()
                rank_size = get_group_size()
                epoch_size_pos = epoch_size_pos // rank_size
                epoch_size_neg = epoch_size_neg // rank_size
            epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations

            which_dataset = list(np.full(epoch_size_pos, fill_value=1))
            which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
            shuffle(which_dataset)
            print("1 epoch = " + str(epoch_size) + " iterations")
            train(net, net_action, net_score, iteration_range,
                  which_dataset, batch_iterator_pos, batch_iterator_neg, all_iteration,
                  dataloader_pos, dataloader_neg)

        t1_wholetracking = time.time()
        t_sum += t1_wholetracking - t0_wholetracking
        print('whole tracking time = %.4f sec.' % (t1_wholetracking - t0_wholetracking))

    # evaluate the precision
    if not args.distributed or rank_id == 0:
        bboxes = np.array(bboxes)
        vid_info['gt'] = np.array(vid_info['gt'])
        if args.run_online == 'True':
            import moxing
            np.save(args.save_result_npy + '-bboxes.npy', bboxes)
            np.save(args.save_result_npy + '-ground_truth.npy', vid_info['gt'])
            moxing.file.copy_parallel('/cache/result', args.train_url)
        else:
            np.save(args.save_result_npy + '-bboxes.npy', bboxes)
            np.save(args.save_result_npy + '-ground_truth.npy', vid_info['gt'])
    return bboxes, t_sum


def redetection(net, curr_bbox_old, opts, cont_negatives, frame, args, frame_idx):
    print('redetection')
    transform = ADNet_Augmentation(opts)
    # redetection process
    redet_samples = gen_samples('gaussian', curr_bbox_old, opts['redet_samples'], opts,
                                min(1.5, 0.6 * 1.15 ** cont_negatives), opts['redet_scale_factor'])
    score_samples = []

    for redet_sample in redet_samples:
        temp_patch, _, _, _ = transform(frame, redet_sample, None, None)
        temp_patch = Tensor(np.expand_dims(temp_patch, 0), mstype.float32).transpose(0, 3, 1, 2)

        # 1 batch input [1, curr_patch.shape]
        _, fc7_out_temp = net.construct(temp_patch, -1, False)
        score_samples.append(fc7_out_temp.asnumpy()[0][1])

    score_samples = np.array(score_samples)
    max_score_samples_idx = np.argmax(score_samples)

    # replace the curr_box with the samples with maximum score
    curr_bbox = redet_samples[max_score_samples_idx]

    # update the final result image
    im_with_bb = display_result(frame, curr_bbox) if args.display_images else draw_box(frame, curr_bbox)

    save_img(args, os.path.join(args.save_result_images, str(frame_idx) + '-redet.jpg'), im_with_bb)


def train(net, net_action, net_score, iteration_range, which_dataset,
          batch_iterator_pos, batch_iterator_neg, all_iteration,
          dataloader_pos, dataloader_neg):
    net.set_phase('train')

    # training loop
    for iteration in iteration_range:
        all_iteration += 1  # use this for update the visualization
        # load train data
        if which_dataset[iteration % len(which_dataset)]:  # if positive
            try:
                images, _, action_label, score_label = next(batch_iterator_pos)
            except StopIteration:
                batch_iterator_pos = iter(dataloader_pos)
                images, _, action_label, score_label = next(batch_iterator_pos)
        else:
            try:
                images, _, action_label, score_label = next(batch_iterator_neg)
            except StopIteration:
                batch_iterator_neg = iter(dataloader_neg)
                images, _, action_label, score_label = next(batch_iterator_neg)

        images = images.transpose(0, 3, 1, 2)
        # forward
        t0 = time.time()
        if which_dataset[iteration % len(which_dataset)]:  # if positive
            action_l = net_action(images, ops.Argmax(1, output_type=mstype.int32)(action_label))
        else:
            action_l = Tensor([0])
        score_l = net_score(images, score_label)
        loss = action_l + score_l
        t1 = time.time()

        if all_iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(all_iteration) + ' || Loss: %.4f ||' % (loss.asnumpy()), end=' ')


def detection(net, opts, args, frame, curr_bbox, cont_negatives, frame_idx, ntraining):
    net.set_phase('test')
    transform = ADNet_Augmentation(opts)
    t = 0
    while True:
        curr_patch, curr_bbox, _, _ = transform(frame, curr_bbox, None, None)
        curr_patch = Tensor(np.expand_dims(curr_patch, 0), mstype.float32).transpose(0, 3, 1, 2)
        fc6_out, fc7_out = net.construct(curr_patch)

        curr_score = fc7_out.asnumpy()[0][1]

        if ntraining > args.believe_score_result:
            if curr_score < opts['failedThre']:
                cont_negatives += 1

        action = np.argmax(fc6_out.asnumpy())

        # do action
        curr_bbox = do_action(curr_bbox, opts, action, frame.shape)

        # bound the curr_bbox size
        if curr_bbox[2] < 10:
            curr_bbox[0] = min(0, curr_bbox[0] + curr_bbox[2] / 2 - 10 / 2)
            curr_bbox[2] = 10
        if curr_bbox[3] < 10:
            curr_bbox[1] = min(0, curr_bbox[1] + curr_bbox[3] / 2 - 10 / 2)
            curr_bbox[3] = 10

        t += 1

        # draw box or with display, then save
        if args.display_images:
            im_with_bb = display_result(frame, curr_bbox)  # draw box and display
        else:
            im_with_bb = draw_box(frame, curr_bbox)
        save_img(args, os.path.join(args.save_result_images, str(frame_idx) + '-' + str(t) + '.jpg'), im_with_bb)

        if action == opts['stop_action'] or t >= opts['num_action_step_max']:
            break
        return curr_score, cont_negatives


def save_img(args, filename, img):
    if args.save_result_images:
        cv2.imwrite(filename, img)


def get_optim(net, loss_fn, opts):
    optimizer = nn.SGD([{'params': net.base_network.trainable_params(), 'lr': 0},
                        {'params': net.fc4_5.trainable_params()},
                        {'params': net.fc6.trainable_params()},
                        {'params': net.fc7.trainable_params(), 'lr': 1e-3}],
                       learning_rate=1e-3, momentum=opts['train']['momentum'],
                       weight_decay=opts['train']['weightDecay'])

    net_action_with_criterion = WithLossCell_ADNET(net, loss_fn, 'action')
    net_score_with_criterion = WithLossCell_ADNET(net, loss_fn, 'score')
    net_action = TrainOneStepCell(net_action_with_criterion, optimizer)
    net_score = TrainOneStepCell(net_score_with_criterion, optimizer)
    return net_action, net_score
