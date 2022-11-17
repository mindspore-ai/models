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

#include "./nn_matching.h"
#include <iostream>

NearNeighborDisMetric::NearNeighborDisMetric(NearNeighborDisMetric::METRIC_TYPE metric,
                                             float matching_threshold, int budget) {
  if (metric == euclidean) {
    _metric = &NearNeighborDisMetric::_nneuclidean_distance;
  } else if (metric == cosine) {
    _metric = &NearNeighborDisMetric::_nncosine_distance;
  }

  this->mating_threshold = matching_threshold;
  this->budget = budget;
  this->samples.clear();
}

DYNAMICM NearNeighborDisMetric::distance(
    const FEATURESS &features,
    const std::vector<int> &targets) {
  DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
  int idx = 0;
  for (int target : targets) {
    cost_matrix.row(idx) = (this->*_metric)(this->samples[target], features);
    idx++;
  }
  return cost_matrix;
}

void NearNeighborDisMetric::partial_fit(
    std::vector<TRACKER_DATA> &tid_feats,
    std::vector<int> &active_targets) {
  for (TRACKER_DATA &data : tid_feats) {
    int track_id = data.first;
    FEATURESS newFeatOne = data.second;
    const int feature_size = 128;

    if (samples.find(track_id) != samples.end()) {  // append
      int oldSize = samples[track_id].rows();
      int addSize = newFeatOne.rows();
      int newSize = oldSize + addSize;

      if (newSize <= this->budget) {
        FEATURESS newSampleFeatures(newSize, feature_size);
        newSampleFeatures.block(0, 0, oldSize, feature_size) = samples[track_id];
        newSampleFeatures.block(oldSize, 0, addSize, feature_size) = newFeatOne;
        samples[track_id] = newSampleFeatures;
      } else {
        // original space is not enough;
        if (oldSize < this->budget) {
          FEATURESS newSampleFeatures(this->budget, feature_size);
          if (addSize >= this->budget) {
            newSampleFeatures = newFeatOne.block(addSize - this->budget, 0, this->budget, feature_size);
          } else {
            newSampleFeatures.block(0, 0, this->budget - addSize, feature_size) =
                samples[track_id].block(newSize - this->budget, 0, this->budget - addSize, feature_size);
            newSampleFeatures.block(this->budget - addSize, 0, addSize, feature_size) = newFeatOne;
          }
          samples[track_id] = newSampleFeatures;
        } else {
          // original space is ok;
          if (addSize >= this->budget) {
            samples[track_id] = newFeatOne.block(addSize - this->budget, 0, this->budget, feature_size);
          } else {
            samples[track_id].block(0, 0, this->budget - addSize, feature_size) =
                samples[track_id].block(newSize - this->budget, 0, this->budget - addSize, feature_size);
            samples[track_id].block(this->budget - addSize, 0, addSize, feature_size) = newFeatOne;
          }
        }
      }
    } else {
      // not exit, create new one;
      samples[track_id] = newFeatOne;
    }
  }  // add features;

  // erase the samples which not in active_targets;
  for (std::map<int, FEATURESS>::iterator i = samples.begin(); i != samples.end();) {
    bool flag = false;
    int f = i->first;
    if (std::any_of(active_targets.begin(), active_targets.end(), [f](int j){return j == f;})) {
      flag = true;
    }

    if (flag == false) {
      samples.erase(i++);
    } else {
      ++i;
    }
  }
}

Eigen::VectorXf NearNeighborDisMetric::_nncosine_distance(
    const FEATURESS &x, const FEATURESS &y) {
  Eigen::MatrixXf distances = _cosine_distance(x, y);
  Eigen::VectorXf res = distances.colwise().minCoeff().transpose();
  return res;
}

Eigen::VectorXf NearNeighborDisMetric::_nneuclidean_distance(const FEATURESS &x, const FEATURESS &y) {
  Eigen::MatrixXf distances = _pdist(x, y);
  Eigen::VectorXf res = distances.colwise().maxCoeff().transpose();
  res = res.array().max(Eigen::VectorXf::Zero(res.rows()).array());
  return res;
}

Eigen::MatrixXf NearNeighborDisMetric::_pdist(const FEATURESS &x, const FEATURESS &y) {
  int len1 = x.rows(), len2 = y.rows();
  if (len1 == 0 || len2 == 0) {
    return Eigen::MatrixXf::Zero(len1, len2);
  }
  Eigen::MatrixXf res = x * y.transpose() * -2;
  res = res.colwise() + x.rowwise().squaredNorm();
  res = res.rowwise() + y.rowwise().squaredNorm().transpose();
  res = res.array().max(Eigen::MatrixXf::Zero(res.rows(), res.cols()).array());
  return res;
}

Eigen::MatrixXf NearNeighborDisMetric::_cosine_distance(
    const FEATURESS &a, const FEATURESS &b, bool data_is_normalized) {
  auto a_tmp = a;
  auto b_tmp = b;
  if (!data_is_normalized) {
    auto a_norm = a_tmp.rowwise().norm();
    auto b_norm = b_tmp.rowwise().norm();
    for (uint32_t id = 0; id < a_norm.rows(); id++) {
      a_tmp.row(id) *= 1.0 / a_norm(id, 0);
    }
    for (uint32_t id = 0; id < b_norm.rows(); id++) {
      b_tmp.row(id) *= 1.0 / b_norm(id, 0);
    }
  }
  Eigen::MatrixXf res = 1. - (a_tmp * b_tmp.transpose()).array();
  return res;
}
