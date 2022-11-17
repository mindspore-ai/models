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
import numpy as np
from matplotlib import pyplot as plt
from src.utils.overlap_ratio import overlap_ratio


def plot_result(Z, title, show=True, save_plot=None, xlabel=None, ylabel=None) -> None:
    plt.plot(Z)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim([0, 1])
    if save_plot:
        plt.savefig(save_plot)
    if show:
        plt.show()

    plt.clf()


def distance_precision_plot(bboxes, ground_truth, title, show=True, save_plot=None):
    # PRECISION_PLOT
    # Calculates precision for a series of distance thresholds (percentage of frames where the distance to the ground
    # truth is within the threshold). The results are shown in a new figure if SHOW is true.

    # Accepts positions and ground truth as Nx2 matrices(for N frames), and a title string.
    # matlab code credit:
    # Joao F.Henriques, 2014
    # http: // www.isr.uc.pt / ~henriques /

    positions = bboxes[:, [1, 0]] + bboxes[:, [3, 2]] / 2
    ground_truth = ground_truth[:, [1, 0]] + ground_truth[:, [3, 2]] / 2

    max_threshold = 50  # used for graphs in the paper

    precisions = np.zeros([max_threshold, 1])

    if len(positions) != len(ground_truth):
        print("WARNING: the size of positions and ground_truth are not same")
        # just ignore any extra frames, in either results or ground truth
        n = min(len(positions), len(ground_truth))
        positions = positions[:n]
        ground_truth = ground_truth[:n]

    # calculate distances to ground truth over all frames
    distances = np.sqrt(
        np.square(positions[:, 0] - ground_truth[:, 0]) + np.square(positions[:, 1] - ground_truth[:, 1]))

    distances = distances[~np.isnan(distances)]

    # compute precision
    precisions = []
    for p in range(max_threshold):
        precisions.append(len(distances[distances <= p]) / len(distances))

    # plot
    if show or save_plot:
        if save_plot is not None:
            save_plot += '-distance'
        plot_result(precisions, title, show=show, save_plot=save_plot, xlabel='distance threshold', ylabel='precision')

    return precisions


def iou_precision_plot(bboxes, ground_truth, title, show=True, save_plot=None):
    max_threshold = 100  # used for graphs in the paper

    # precisions = np.zeros([max_threshold, 1])

    if len(bboxes) != len(ground_truth):
        print("WARNING: the size of iou and ground_truth are not same")
        # just ignore any extra frames, in either results or ground truth
        n = min(len(bboxes), len(ground_truth))
        ground_truth = ground_truth[:n]

    iou = overlap_ratio(bboxes, ground_truth)
    iou = np.array(iou)

    # compute precision
    precisions = []
    for p in range(max_threshold):
        precisions.append(len(iou[iou >= p/100.0]) / len(iou))

    # plot
    if show or save_plot:
        if save_plot is not None:
            save_plot += '-iou'
        plot_result(precisions, title,
                    show=show, save_plot=save_plot, xlabel='iou threshold (x0.01)', ylabel='precision')
    return precisions
