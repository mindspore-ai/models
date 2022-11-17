/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRACKER_H
#define TRACKER_H
#include <vector>


#include "./kalmanfilter.h"
#include "./track.h"
#include "../DeepAppearanceDescriptor/model.h"

class NearNeighborDisMetric;

class tracker {
 public:
    NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;

    KalmanFilter* kf;

    int _next_idx;

 public:
    std::vector<Track> tracks;
    tracker(float max_cosine_distance, int nn_budget,
            float max_iou_distance = 0.7,
            int max_age = 70, int n_init = 3);
    void predict();
    void update(const DETECTIONS& detections);
    typedef DYNAMICM (tracker::* GATED_METRIC_FUNC)(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);

 private:
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);

 public:
    DYNAMICM gated_matric(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox,
            DETECTBOXSS &candidates);
};

#endif  // TRACKER_H
