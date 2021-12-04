/*
 * Copyright 2021 Huawei Technologies Co., Ltd. All rights reserved.
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
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "FairmotMindsporePost.h"

#include <dirent.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>
#include <boost/property_tree/json_parser.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/opencv.hpp>

#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "acl/acl.h"

namespace {
// Output Tensor
const int OUTPUT_TENSOR_SIZE = 2;
const int OUTPUT_ID_FEATURE_INDEX = 0;
const int OUTPUT_DETS_INDEX = 1;

const int OUTPUT_ID_FEATURE_SIZE = 2;
const int OUTPUT_DETS_SIZE = 3;
const float CONF_THRES = 0.3;
const int TRACK_BUFFER = 30;
const int K = 500;
const int TrackState_New = 0;
const int TrackState_Tracked = 1;
const int TrackState_Lost = 2;
const int TrackState_Removed = 3;
const int MIN_BOX_AREA = 100;
}  // namespace

namespace MxBase {
BaseTrack basetrack;
KalmanFilter kalmanfilter;
void FairmotMindsporePost::get_result(int *collist, int cols, cost *d, cost min,
                                      int rows, int &endofpath, row *colsol,
                                      cost **assigncost, cost *v, int *pred) {
  col low = 0, up = 0;  // columns in 0..low-1 are ready, now none.
  int k, j, i, j1;
  cost h, v2;
  boolean unassignedfound = FALSE;
  do {
    if (up == low) {
      min = d[collist[up++]];
      for (k = up; k < cols; k++) {
        j = collist[k];
        h = d[j];
        if (h <= min) {
          if (h < min) {
            up = low;  // restart list at index low.
            min = h;
          }
          collist[k] = collist[up];
          collist[up++] = j;
        }
      }
      for (k = low; k < up; k++)
        if (colsol[collist[k]] < 0) {
          endofpath = collist[k];
          unassignedfound = TRUE;
          break;
        }
    }
    if (!unassignedfound) {
      j1 = collist[low];
      low++;
      i = colsol[j1];
      if (i > rows) {
        i = 0;
      }
      h = assigncost[i][j1] - v[j1] - min;
      for (k = up; k < cols; k++) {
        j = collist[k];
        v2 = assigncost[i][j] - v[j] - h;
        if (v2 < d[j]) {
          pred[j] = i;
          if (v2 == min) {
            if (colsol[j] < 0) {
              endofpath = j;
              unassignedfound = TRUE;
              break;
            } else {
              collist[k] = collist[up];
              collist[up++] = j;
            }
          }
          d[j] = v2;
        }
      }
    }
  } while (!unassignedfound);
}
void FairmotMindsporePost::func(int &numfree, int *free, cost **assigncost,
                                row *colsol, cost *v, col *rowsol, int cols) {
  int j2, loopcnt = 0;  // do-loop to be done twice.
  do {
    loopcnt++;
    int k = 0;
    row prvnumfree;
    prvnumfree = numfree;
    numfree = 0;
    while (k < prvnumfree) {
      int i = free[k++];
      int umin = assigncost[i][0] - v[0];
      int j1 = 0;
      int usubmin = BIG;
      for (int j = 1; j < cols; j++) {
        int h = assigncost[i][j] - v[j];
        if (h < usubmin) {
          if (h >= umin) {
            usubmin = h;
            j2 = j;
          } else {
            usubmin = umin;
            umin = h;
            j2 = j1;
            j1 = j;
          }
        }
      }
      int i0 = colsol[j1];
      if (umin < usubmin) {
        v[j1] = v[j1] - (usubmin - umin);
      } else {
        if (i0 > -1) {
          j1 = j2;
          i0 = colsol[j2];
        }
      }
      rowsol[i] = j1;
      colsol[j1] = i;
      if (i0 > -1) {
        if (umin < usubmin) {
          free[--k] = i0;
        } else {
          free[numfree++] = i0;
        }
      }
    }
  } while (loopcnt < 2);  // repeat once.
}

void FairmotMindsporePost::lap(cost **assigncost, col *rowsol, row *colsol,
                               cost *u, cost *v, int rows, int cols) {
  int i, numfree = 0, f, *pred = new row[rows], *free = new row[rows], j, j1,
         endofpath, *collist = new col[cols], *matches = new col[cols];
  cost min, *d = new cost[rows];
  for (i = 0; i < rows; i++) matches[i] = 0;
  for (j = cols; j--;) {  // reverse order gives better results.
    row imin = 0;
    min = assigncost[0][j];
    for (i = 1; i < rows; i++)
      if (assigncost[i][j] < min) {
        min = assigncost[i][j];
        imin = i;
      }
    v[j] = min;
    if (++matches[imin] == 1) {
      rowsol[imin] = j;
      colsol[j] = imin;
    } else if (v[j] < v[rowsol[imin]]) {
      int j_1 = rowsol[imin];
      rowsol[imin] = j;
      colsol[j] = imin;
      colsol[j_1] = -1;
    } else {
      colsol[j] = -1;
    }
  }
  for (i = 0; i < rows; i++)
    if (matches[i] == 0) {
      free[numfree++] = i;
    } else if (matches[i] == 1) {
      j1 = rowsol[i];
      min = BIG;
      for (j = 0; j < cols; j++)
        if (j != j1)
          if (assigncost[i][j] - v[j] < min) min = assigncost[i][j] - v[j];
      v[j1] = v[j1] - min;
    }
  func(numfree, free, assigncost, colsol, v, rowsol, cols);
  for (f = 0; f < numfree; f++) {
    row freerow;
    freerow = free[f];  // start row of augmenting path.
    for (j = cols; j--;) {
      d[j] = assigncost[freerow][j] - v[j];
      pred[j] = freerow;
      collist[j] = j;  // init column list.
    }
    get_result(collist, cols, d, min, rows, endofpath, colsol, assigncost, v,
               pred);
    do {
      i = pred[endofpath];
      colsol[endofpath] = i;
      j1 = endofpath;
      endofpath = rowsol[i];
      rowsol[i] = j1;
    } while (i != freerow);
  }
  delete[] matches;
}

int BaseTrack::next_id() {
  this->count += 1;
  return this->count;
}

Results::Results(uint32_t frame_id, const std::vector<cv::Mat> &online_tlwhs,
                 const std::vector<int> &online_ids) {
  this->frame_id = frame_id;
  this->online_tlwhs = online_tlwhs;
  this->online_ids = online_ids;
}

JDETracker::JDETracker(uint32_t frame_rate) {
  this->det_thresh = CONF_THRES;
  this->buffer_size = static_cast<int>(frame_rate / 30.0 * TRACK_BUFFER);
  this->max_time_lost = this->buffer_size;
  this->max_per_image = K;
  KalmanFilter kalman_filter;
  this->kalman_filter = kalman_filter;
}
STack::STack() {}
STack::STack(cv::Mat tlwh, float score, cv::Mat temp_feat,
             uint32_t buffer_size) {
  tlwh.convertTo(this->tlwh, CV_64FC1);
  this->is_activated = false;
  this->score = score;
  this->tracklet_len = 0;
  this->update_features(temp_feat);
  this->alpha = 0.9;
}
cv::Mat STack::gettlwh() {
  if (this->mean.rows == 0) {
    return this->tlwh.clone();
  } else {
    cv::Mat ret =
        this->mean(cv::Range(0, this->mean.rows), cv::Range(0, 4)).clone();
    ret.at<double>(0, 2) *= ret.at<double>(0, 3);
    for (size_t i = 0; i < 2; i++) {
      ret.at<double>(0, i) -= ret.at<double>(0, i + 2) / 2;
    }
    return ret;
  }
}
cv::Mat STack::tlbr() {
  cv::Mat ret = this->gettlwh();
  for (size_t i = 0; i < 2; i++) {
    ret.at<double>(0, i + 2) += ret.at<double>(0, i);
  }
  return ret;
}
cv::Mat STack::tlwh_to_xyah(cv::Mat tlwh) {
  cv::Mat ret = tlwh;
  for (size_t i = 0; i < 2; i++) {
    ret.at<double>(0, i) += ret.at<double>(0, i + 2) / 2;
  }
  ret.at<double>(0, 2) /= ret.at<double>(0, 3);
  return ret;
}
void STack::activate(const KalmanFilter &kalman_filter, uint32_t frame_id) {
  this->kalman_filter = kalman_filter;
  this->track_id = basetrack.next_id();
  this->kalman_filter.initiate(this->tlwh_to_xyah(this->tlwh), this->mean,
                               this->covariance);
  this->tracklet_len = 0;
  this->state = TrackState_Tracked;
  this->is_activated = false;
  if (frame_id == 1) {
    this->is_activated = true;
  }
  this->frame_id = frame_id;
  this->start_frame = frame_id;
}
void STack::update_features(cv::Mat temp_feat) {
  cv::Mat feat;
  cv::normalize(temp_feat, feat);
  this->curr_feat = feat;
  if (this->smooth_feat.empty()) {
    this->smooth_feat = feat;
  } else {
    this->smooth_feat =
        this->alpha * this->smooth_feat + (1 - this->alpha) * feat;
  }
  this->features.push_back(feat);
  cv::normalize(this->smooth_feat, this->smooth_feat);
}

void STack::update(STack new_track, int frame_id, bool update_feature) {
  this->frame_id = frame_id;
  this->tracklet_len += 1;
  cv::Mat new_tlwh = new_track.gettlwh();
  this->kalman_filter.update(this->mean, this->covariance,
                             this->tlwh_to_xyah(new_tlwh));
  this->state = TrackState_Tracked;
  this->is_activated = true;
  this->score = new_track.score;
  if (update_feature == true) {
    this->update_features(new_track.curr_feat);
  }
}

void STack::re_activate(STack new_track, int frame_id, bool new_id) {
  this->kalman_filter.update(this->mean, this->covariance,
                             this->tlwh_to_xyah(new_track.gettlwh()));
  this->update_features(new_track.curr_feat);
  this->tracklet_len = 0;
  this->state = TrackState_Tracked;
  this->is_activated = true;
  this->frame_id = frame_id;
  if (new_id) {
    this->track_id = basetrack.next_id();
  }
}

void KalmanFilter::initiate(cv::Mat measurement, cv::Mat &mean,
                            cv::Mat &covariance) {
  cv::Mat mean_pos = measurement;
  cv::Mat mean_vel = cv::Mat::zeros(mean_pos.rows, mean_pos.cols, CV_64FC1);
  hconcat(mean_pos, mean_vel, mean);
  double tmp[1][8] = {
      2 * this->std_weight_position * measurement.at<double>(0, 3),
      2 * this->std_weight_position * measurement.at<double>(0, 3),
      1e-2,
      2 * this->std_weight_position * measurement.at<double>(0, 3),
      10 * this->std_weight_velocity * measurement.at<double>(0, 3),
      10 * this->std_weight_velocity * measurement.at<double>(0, 3),
      1e-5,
      10 * this->std_weight_velocity * measurement.at<double>(0, 3)};
  cv::Mat std = cv::Mat(1, 8, CV_64FC1, tmp);
  std = std.mul(std);
  covariance = cv::Mat::eye(std.cols, std.cols, CV_64FC1);
  for (size_t i = 0; i < std.cols; i++)
    covariance.at<double>(i, i) = std.at<double>(0, i);
}
void KalmanFilter::multi_predict(cv::Mat &mean,
                                 std::vector<cv::Mat> &covariance) {
  cv::Mat std_pos(4, mean.rows, CV_64FC1);
  cv::Mat std_vel(4, mean.rows, CV_64FC1);
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < mean.rows; j++)
      if (i == 2) {
        std_pos.at<double>(i, j) = 1e-2;
        std_vel.at<double>(i, j) = 1e-5;
      } else {
        std_pos.at<double>(i, j) =
            this->std_weight_position * mean.at<double>(j, 3);
        std_vel.at<double>(i, j) =
            this->std_weight_velocity * mean.at<double>(j, 3);
      }
  cv::Mat sqr;
  vconcat(std_pos, std_vel, sqr);
  sqr = sqr.mul(sqr).t();
  std::vector<cv::Mat> motion_cov;
  for (size_t i = 0; i < mean.rows; i++) {
    cv::Mat diag = cv::Mat::eye(sqr.cols, sqr.cols, CV_64FC1);
    for (size_t j = 0; j < sqr.cols; j++) {
      diag.at<double>(j, j) = sqr.at<double>(i, j);
    }
    motion_cov.push_back(diag);
  }
  mean = mean * this->motion_mat.t();
  std::vector<cv::Mat> left;
  for (size_t i = 0; i < covariance.size(); i++) {
    left.push_back(this->motion_mat * covariance[i]);
  }
  for (size_t i = 0; i < covariance.size(); i++) {
    covariance[i] = left[i] * this->motion_mat.t() + motion_cov[i];
  }
}
KalmanFilter::KalmanFilter() {
  this->ndim = 4;
  this->dt = 1.0;
  this->motion_mat = cv::Mat::eye(2 * (this->ndim), 2 * (this->ndim), CV_64FC1);
  for (size_t i = 0; i < this->ndim; ++i) {
    this->motion_mat.at<double>(i, this->ndim + i) = this->dt;
  }
  this->update_mat = cv::Mat::eye(this->ndim, 2 * (this->ndim), CV_64FC1);
  this->std_weight_position = 1.0 / 20;
  this->std_weight_velocity = 1.0 / 160;
}
cv::Mat KalmanFilter::GatingDistance(cv::Mat mean, cv::Mat covariance,
                                     cv::Mat measurements, bool only_position,
                                     const std::string &metric) {
  this->project(mean, covariance);
  cv::Mat d(measurements.rows, measurements.cols, CV_64FC1);
  for (size_t i = 0; i < measurements.rows; i++) {
    d.row(i) = measurements.row(i) - mean;
  }
  cv::Mat cholesky_factor = cv::Mat::zeros(covariance.size(), CV_64F);
  for (int i = 0; i < covariance.rows; ++i) {
    int j;
    double sum;
    for (j = 0; j < i; ++j) {
      sum = 0;
      for (int k = 0; k < j; ++k) {
        sum +=
            cholesky_factor.at<double>(i, k) * cholesky_factor.at<double>(j, k);
      }
      cholesky_factor.at<double>(i, j) = (covariance.at<double>(i, j) - sum) /
                                         cholesky_factor.at<double>(j, j);
    }
    sum = 0;
    assert(i == j);
    for (int k = 0; k < j; ++k) {
      sum +=
          cholesky_factor.at<double>(j, k) * cholesky_factor.at<double>(j, k);
    }
    cholesky_factor.at<double>(j, j) = sqrt(covariance.at<double>(j, j) - sum);
  }
  cv::Mat z;
  cv::solve(cholesky_factor, d.t(), z);
  z = z.mul(z);
  cv::Mat squared_maha(1, z.cols, CV_64FC1);
  for (size_t i = 0; i < z.cols; i++) {
    double sum = 0;
    for (size_t t = 0; t < z.rows; t++) {
      sum += z.at<double>(t, i);
    }
    squared_maha.at<double>(0, i) = sum;
  }
  return squared_maha;
}

void KalmanFilter::project(cv::Mat &mean, cv::Mat &covariance) {
  cv::Mat std(1, 4, CV_64FC1);
  for (size_t i = 0; i < 4; i++)
    if (i == 2)
      std.at<double>(0, i) = 1e-1;
    else
      std.at<double>(0, i) = this->std_weight_position * mean.at<double>(0, 3);
  std = std.mul(std);
  cv::Mat innovation_cov = cv::Mat::eye(std.cols, std.cols, CV_64FC1);
  for (size_t j = 0; j < std.cols; j++) {
    innovation_cov.at<double>(j, j) = std.at<double>(0, j);
  }
  cv::Mat tmp(mean.rows, this->update_mat.rows, CV_64FC1);
  for (size_t i = 0; i < this->update_mat.rows; i++) {
    tmp.at<double>(0, i) = this->update_mat.row(i).dot(mean);
  }
  mean = tmp;
  covariance =
      this->update_mat * covariance * this->update_mat.t() + innovation_cov;
}

void KalmanFilter::update(cv::Mat &mean, cv::Mat &covariance,
                          cv::Mat measurement) {
  cv::Mat projected_mean = mean.clone();
  cv::Mat projected_cov = covariance.clone();
  this->project(projected_mean, projected_cov);
  cv::Mat chol_factor;
  this->cholesky_decomposition(projected_cov, chol_factor);
  cv::Mat b = covariance * this->update_mat.t();
  b = b.t();
  cv::Mat kalman_gain(chol_factor.rows, b.cols, CV_64FC1);
  for (size_t i = 0; i < b.cols; i++) {
    f64_vec_t x = {0, 0, 0, 0};
    f64_vec_t *f_x = &x;
    cv::Mat mat_b(b.rows, 1, CV_64FC1);
    mat_b = b.col(i);
    this->chol_subtitute(chol_factor, mat_b, f_x, chol_factor.rows);
    for (size_t j = 0; j < chol_factor.rows; j++) {
      if ((*f_x)[j] < 0.001)
        kalman_gain.at<double>(j, i) = 0;
      else
        kalman_gain.at<double>(j, i) = (*f_x)[j];
    }
  }
  kalman_gain = kalman_gain.t();
  cv::Mat innovation = measurement - projected_mean;
  mean += innovation * kalman_gain.t();
  covariance -= kalman_gain * projected_cov * kalman_gain.t();
}

void KalmanFilter::chol_subtitute(cv::Mat chol_factor, cv::Mat mat_b,
                                  f64_vec_t *f_x, int n) {
  f64_mat_t L;
  f64_vec_t b;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) L[i][j] = chol_factor.at<double>(i, j);
  for (int i = 0; i < n; ++i) b[i] = mat_b.at<double>(i, 0);
  f64_mat_t *f_L = &L;
  f64_vec_t *f_b = &b;
  int i, j;
  double f_sum;

  double *pX;
  double *pXj;
  const double *pL;
  const double *pB;
  /** @llr - This function shall solve the unknown vector f_x given a matrix f_L
   * and a vector f_b with the relation f_L*fL'*f_x = f_b.*/

  pX = &(*f_x)[0];
  pB = &(*f_b)[0];
  /* Copy f_b into f_x */
  for (i = 0u; i < n; i++) {
    (*pX) = (*pB);
    pX++;
    pB++;
  }
  /* Solve Ly = b  for y */
  pXj = &(*f_x)[0];
  for (i = 0u; i < n; i++) {
    double *pXi;
    double fLii;
    f_sum = (*f_x)[i];
    fLii = (*f_L)[i][i];
    pXi = &(*f_x)[0];
    pL = &(*f_L)[i][0];

    for (j = 0u; j < i; j++) {
      f_sum -= (*pL) * (*pXi);
      pL++;
      pXi++;
    }
    (*pXj) = f_sum / fLii;
    pXj++;
  }
  /* Solve L'x = y for x */
  for (i = 1u; i <= n; i++) {
    f_sum = (*f_x)[n - i];
    pXj = &(*f_x)[n - i + 1u];
    pL = &(*f_L)[n - i + 1u][n - i];
    for (j = n - i + 1u; j < n; j++) {
      f_sum -= (*pL) * (*pXj);
      pXj++;
      pL += n; /* PRQA S 0488 */
    }
    (*f_x)[n - i] = f_sum / (*f_L)[n - i][n - i];
  }
}

void KalmanFilter::cholesky_decomposition(const cv::Mat &A, cv::Mat &L) {
  L = cv::Mat::zeros(A.size(), CV_64F);
  int rows = A.rows;

  for (int i = 0; i < rows; ++i) {
    int j;
    float sum;

    for (j = 0; j < i; ++j) {
      sum = 0;
      for (int k = 0; k < j; ++k) {
        sum += L.at<double>(i, k) * L.at<double>(j, k);
      }
      L.at<double>(i, j) = (A.at<double>(i, j) - sum) / L.at<double>(j, j);
    }
    sum = 0;
    assert(i == j);
    for (int k = 0; k < j; ++k) {
      sum += L.at<double>(j, k) * L.at<double>(j, k);
    }
    L.at<double>(j, j) = sqrt(A.at<double>(j, j) - sum);
  }
}

FairmotMindsporePost &FairmotMindsporePost::operator=(
    const FairmotMindsporePost &other) {
  if (this == &other) {
    return *this;
  }
  PostProcessBase::operator=(other);
  return *this;
}

APP_ERROR FairmotMindsporePost::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
  LogInfo << "Begin to initialize FairmotMindsporePost.";
  APP_ERROR ret = PostProcessBase::Init(postConfig);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Fail to superinit  in PostProcessBase.";
    return ret;
  }

  LogInfo << "End to initialize FairmotMindsporePost.";
  return APP_ERR_OK;
}

APP_ERROR FairmotMindsporePost::DeInit() {
  LogInfo << "Begin to deinitialize FairmotMindsporePost.";
  LogInfo << "End to deinitialize FairmotMindsporePost.";
  return APP_ERR_OK;
}

bool FairmotMindsporePost::IsValidTensors(
    const std::vector<TensorBase> &tensors) const {
  if (tensors.size() < OUTPUT_TENSOR_SIZE) {
    LogError << "The number of tensor (" << tensors.size()
             << ") is less than required (" << OUTPUT_TENSOR_SIZE << ")";
    return false;
  }

  auto idFeatureShape = tensors[OUTPUT_ID_FEATURE_INDEX].GetShape();
  if (idFeatureShape.size() != OUTPUT_ID_FEATURE_SIZE) {
    LogError << "The number of tensor[" << OUTPUT_ID_FEATURE_INDEX
             << "] dimensions (" << idFeatureShape.size()
             << ") is not equal to (" << OUTPUT_ID_FEATURE_SIZE << ")";
    return false;
  }

  auto detsShape = tensors[OUTPUT_DETS_INDEX].GetShape();
  if (detsShape.size() != OUTPUT_DETS_SIZE) {
    LogError << "The number of tensor[" << OUTPUT_DETS_INDEX << "] dimensions ("
             << detsShape.size() << ") is not equal to ("
             << OUTPUT_ID_FEATURE_SIZE << ")";
    return false;
  }

  return true;
}

void FairmotMindsporePost::TransformPreds(const cv::Mat &coords,
                                          MxBase::JDETracker tracker,
                                          cv::Mat &target_coords) {
  target_coords = cv::Mat::zeros(coords.rows, coords.cols, CV_32FC1);
  float scale = tracker.s;
  uint32_t h = tracker.out_height;
  uint32_t w = tracker.out_width;
  float scale_value[1][2] = {scale, scale};
  cv::Mat scale_tmp = cv::Mat(1, 2, CV_32FC1, scale_value);
  float src_w = scale_tmp.at<float>(0, 0);
  uint32_t dst_w = w;
  uint32_t dst_h = h;
  float sn = 0.0;
  float cs = 1.0;
  uint32_t src_point_0 = 0;
  float src_point_1 = src_w * (-0.5);
  float src_dir_value[1][2] = {src_point_0 * cs - src_point_1 * sn,
                               src_point_0 * sn + src_point_1 * cs};
  cv::Mat src_dir = cv::Mat(1, 2, CV_32FC1, src_dir_value);
  float dst_dir_value[1][2] = {0, static_cast<float>(dst_w * (-0.5))};
  cv::Mat dst_dir = cv::Mat(1, 2, CV_32FC1, dst_dir_value);
  cv::Mat src = cv::Mat::zeros(3, 2, CV_32FC1);
  cv::Mat dst = cv::Mat::zeros(3, 2, CV_32FC1);
  float center_value[1][2] = {tracker.c[0], tracker.c[1]};
  cv::Mat center = cv::Mat(1, 2, CV_32FC1, center_value);
  cv::Mat shift = cv::Mat::zeros(1, 2, CV_32FC1);
  cv::Mat src_0 = scale_tmp.mul(shift) + center;
  cv::Mat src_1 = scale_tmp.mul(shift) + center + src_dir;
  cv::Mat direct = src_0 - src_1;
  float direct_tmp_value[1][2] = {-direct.at<float>(0, 1),
                                  direct.at<float>(0, 0)};
  cv::Mat direct_tmp = cv::Mat(1, 2, CV_32FC1, direct_tmp_value);
  cv::Mat src_2 = src_1 + direct_tmp;
  float dst_0_value[1][2] = {static_cast<float>(dst_w * (0.5)),
                             static_cast<float>(dst_h * (0.5))};
  cv::Mat dst_0 = cv::Mat(1, 2, CV_32FC1, dst_0_value);
  float dst_1_value[1][2] = {
      static_cast<float>(dst_w * (0.5)) + dst_dir.at<float>(0, 0),
      static_cast<float>(dst_h * (0.5)) + dst_dir.at<float>(0, 1)};
  cv::Mat dst_1 = cv::Mat(1, 2, CV_32FC1, dst_1_value);
  direct = dst_0 - dst_1;
  float direct_value[1][2] = {-direct.at<float>(0, 1), direct.at<float>(0, 0)};
  direct_tmp = cv::Mat(1, 2, CV_32FC1, direct_value);
  cv::Mat dst_2 = dst_1 + direct_tmp;
  for (size_t y = 0; y < src.cols; ++y) {
    src.at<float>(0, y) = src_0.at<float>(0, y);
    src.at<float>(1, y) = src_1.at<float>(0, y);
    src.at<float>(2, y) = src_2.at<float>(0, y);
    dst.at<float>(0, y) = dst_0.at<float>(0, y);
    dst.at<float>(1, y) = dst_1.at<float>(0, y);
    dst.at<float>(2, y) = dst_2.at<float>(0, y);
  }
  cv::Mat trans(2, 3, CV_32FC1);
  trans = cv::getAffineTransform(dst, src);
  trans.convertTo(trans, CV_32F);
  for (size_t x = 0; x < coords.rows; ++x) {
    float pt_value[3][1] = {coords.at<float>(x, 0), coords.at<float>(x, 1),
                            1.0};
    cv::Mat pt = cv::Mat(3, 1, CV_32FC1, pt_value);
    for (size_t y = 0; y < 2; ++y) {
      cv::Mat new_pt = trans * pt;
      target_coords.at<float>(x, y) = new_pt.at<float>(y);
    }
  }
}

void FairmotMindsporePost::PostProcess(cv::Mat &det,
                                       const MxBase::JDETracker &tracker) {
  cv::Mat coords_0 = det(cv::Range(0, det.rows), cv::Range(0, 2));
  cv::Mat coords_1 = det(cv::Range(0, det.rows), cv::Range(2, 4));
  cv::Mat target_coords_0;
  cv::Mat target_coords_1;
  TransformPreds(coords_0, tracker, target_coords_0);
  TransformPreds(coords_1, tracker, target_coords_1);
  for (size_t x = 0; x < det.rows; ++x) {
    for (size_t y = 0; y < 4; ++y) {
      if (y < 2) {
        det.at<float>(x, y) = target_coords_0.at<float>(x, y);
      } else {
        det.at<float>(x, y) = target_coords_1.at<float>(x, y - 2);
      }
    }
  }
  det = det(cv::Range(0, det.rows), cv::Range(0, 5));
}
void FairmotMindsporePost::TensorBaseToCVMat(cv::Mat &imageMat,
                                             const MxBase::TensorBase &tensor) {
  TensorBase Data = tensor;
  uint32_t outputModelWidth;
  uint32_t outputModelHeight;
  auto shape = Data.GetShape();
  if (shape.size() == 2) {
    outputModelWidth = shape[0];
    outputModelHeight = shape[1];
  } else {
    outputModelWidth = shape[1];
    outputModelHeight = shape[2];
  }
  auto *data = reinterpret_cast<float *>(GetBuffer(Data, 0));
  cv::Mat dataMat(outputModelWidth, outputModelHeight, CV_32FC1);
  for (size_t x = 0; x < outputModelWidth; ++x) {
    for (size_t y = 0; y < outputModelHeight; ++y) {
      dataMat.at<float>(x, y) = data[x * outputModelHeight + y];
    }
  }
  imageMat = dataMat.clone();
}

std::vector<cv::Mat> FairmotMindsporePost::get_lap(cost **assigncost,
                                                   col *rowsol, row *colsol,
                                                   cost *u, cost *v, int row,
                                                   int col, int dim) {
  std::vector<cv::Mat> results;
  lap(assigncost, rowsol, colsol, u, v, dim, dim);
  cv::Mat x(1, row, CV_32FC1), y(1, col, CV_32FC1);
  for (int i = 0; i < row; i++)
    x.at<float>(0, i) = rowsol[i] > (col - 1) ? (-1) : rowsol[i];
  for (int j = 0; j < col; j++)
    y.at<float>(0, j) = colsol[j] > (row - 1) ? (-1) : colsol[j];
  cv::Mat matches(0, 2, CV_32FC1);
  for (size_t i = 0; i < x.cols; i++) {
    if (x.at<float>(0, i) >= 0) {
      cv::Mat tmp(1, 2, CV_32FC1);
      tmp.at<float>(0, 0) = i;
      tmp.at<float>(0, 1) = x.at<float>(0, i);
      matches.push_back(tmp);
    }
  }
  std::vector<int> a, b;
  for (size_t i = 0; i < x.cols; i++)
    if (x.at<float>(0, i) < 0) a.push_back(i);
  for (size_t i = 0; i < y.cols; i++)
    if (y.at<float>(0, i) < 0) b.push_back(i);
  cv::Mat unmatched_a(1, a.size(), CV_32FC1),
      unmatched_b(1, b.size(), CV_32FC1);
  for (size_t i = 0; i < a.size(); i++) unmatched_a.at<float>(0, i) = a[i];
  for (size_t i = 0; i < b.size(); i++) unmatched_b.at<float>(0, i) = b[i];
  results.push_back(matches);
  results.push_back(unmatched_a);
  results.push_back(unmatched_b);
  return results;
}

std::vector<cv::Mat> FairmotMindsporePost::LinearAssignment(cv::Mat cost_matrix,
                                                            float thresh) {
  if (cost_matrix.rows == 0) {
    std::vector<cv::Mat> results;
    cv::Mat matches(0, 2, CV_32FC1), u_track,
        u_detection(1, cost_matrix.cols, CV_32FC1);
    for (size_t i = 0; i < cost_matrix.cols; ++i)
      u_detection.at<float>(0, i) = i;
    results.push_back(matches);
    results.push_back(u_track);
    results.push_back(u_detection);
    return results;
  } else {
    int row = cost_matrix.rows, col = cost_matrix.cols;
    int N = row > col ? row : col;
    cv::Mat cost_c_extended = cv::Mat::ones(2 * N, 2 * N, CV_64FC1);
    cost_c_extended *= thresh;
    if (row != col) {
      double min = 0, max = 0;
      double *minp = &min, *maxp = &max;
      cv::minMaxIdx(cost_matrix, minp, maxp);
      for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
          cost_c_extended.at<double>(i, j) = (*maxp) + thresh + 1;
    }
    for (size_t i = 0; i < row; i++)
      for (size_t j = 0; j < col; j++)
        cost_c_extended.at<double>(i, j) = cost_matrix.at<double>(i, j);
    cost_matrix = cost_c_extended;
    int dim = 2 * N, *rowsol = new int[dim], *colsol = new int[dim];
    double **costMatrix = new double *[dim], *u = new double[dim],
           *v = new double[dim];
    for (int i = 0; i < dim; i++) costMatrix[i] = new double[dim];
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        costMatrix[i][j] = cost_matrix.at<double>(i, j);
    return get_lap(costMatrix, rowsol, colsol, u, v, row, col, dim);
  }
}

void FairmotMindsporePost::FuseMotion(MxBase::KalmanFilter &kf,
                                      cv::Mat &cost_matrix,
                                      std::vector<STack *> tracks,
                                      std::vector<STack *> detections,
                                      bool only_position, float lambda_) {
  if (cost_matrix.rows != 0) {
    int gating_dim;
    if (only_position = false)
      gating_dim = 2;
    else
      gating_dim = 4;
    float gating_threshold = kalmanfilter.chi2inv95[gating_dim];
    cv::Mat measurements(detections.size(), 4, CV_64FC1);
    for (size_t i = 0; i < detections.size(); i++)
      measurements.row(i) =
          (*detections[i]).tlwh_to_xyah((*detections[i]).gettlwh()) + 0;
    for (size_t i = 0; i < tracks.size(); i++) {
      cv::Mat gating_distance =
          kf.GatingDistance((*tracks[i]).mean, (*tracks[i]).covariance,
                            measurements, only_position);
      for (size_t t = 0; t < gating_distance.cols; t++)
        if (gating_distance.at<double>(0, t) > gating_threshold)
          cost_matrix.at<double>(i, t) = DBL_MAX;
      cost_matrix.row(i) =
          lambda_ * cost_matrix.row(i) + (1 - lambda_) * gating_distance;
    }
  }
}
std::vector<STack *> FairmotMindsporePost::JointStracks(
    std::vector<STack *> tlista, std::vector<STack *> tlistb) {
  std::vector<STack *> res;
  std::map<int, int> exists;
  for (size_t t = 0; t < tlista.size(); t++) {
    exists[(*tlista[t]).track_id] = 1;
    res.push_back(tlista[t]);
  }
  for (size_t t = 0; t < tlistb.size(); t++) {
    int tid = (*tlistb[t]).track_id;
    if (exists[tid] == 0) {
      exists[tid] = 1;
      res.push_back(tlistb[t]);
    }
  }
  return res;
}
void FairmotMindsporePost::RemoveDuplicateStracks(
    std::vector<STack *> &stracksa, std::vector<STack *> &stracksb) {
  cv::Mat pdist = IouDistance(stracksa, stracksb);
  std::vector<size_t> p, q, dupa, dupb;
  std::vector<STack *> resa;
  std::vector<STack *> resb;
  for (size_t i = 0; i < pdist.rows; i++)
    for (size_t j = 0; j < pdist.cols; j++)
      if (pdist.at<double>(i, j) < 0.15) {
        p.push_back(i);
        q.push_back(j);
      }
  for (size_t i = 0; i < p.size(); i++) {
    int timep = (*stracksa[p[i]]).frame_id - (*stracksa[p[i]]).start_frame;
    int timeq = (*stracksb[q[i]]).frame_id - (*stracksb[q[i]]).start_frame;
    if (timep > timeq) {
      dupb.push_back(q[i]);
    } else {
      dupa.push_back(p[i]);
    }
  }
  for (size_t i = 0; i < stracksa.size(); i++) {
    if (std::find(dupa.begin(), dupa.end(), i) == dupa.end()) {
      resa.push_back(stracksa[i]);
    }
  }
  for (size_t i = 0; i < stracksb.size(); i++) {
    if (std::find(dupb.begin(), dupb.end(), i) == dupb.end()) {
      resb.push_back(stracksb[i]);
    }
  }
  stracksa = resa;
  stracksb = resb;
}
std::vector<STack *> FairmotMindsporePost::SubStracks(
    std::vector<STack *> tlista, std::vector<STack *> tlistb) {
  std::vector<STack *> res;
  std::map<size_t, STack *> stracks;
  std::map<size_t, STack *>::iterator it;
  std::vector<size_t> key;
  std::vector<size_t> del_key;
  for (size_t t = 0; t < tlista.size(); t++) {
    key.push_back((*tlista[t]).track_id);
    stracks[(*tlista[t]).track_id] = tlista[t];
  }
  for (size_t t = 0; t < tlistb.size(); t++) {
    int tid = (*tlistb[t]).track_id;
    it = stracks.find(tid);
    if (it != stracks.end()) {
      del_key.push_back(tid);
      stracks.erase(it);
    }
  }
  for (size_t i = 0; i < key.size(); i++) {
    bool flag = false;
    for (size_t j = 0; j < del_key.size(); j++) {
      if (del_key[j] == key[i]) {
        flag = true;
      }
      if (flag == true) {
        break;
      }
    }
    if (flag == false) {
      res.push_back(stracks[key[i]]);
    }
  }
  return res;
}
cv::Mat FairmotMindsporePost::BboxOverlaps(std::vector<cv::Mat> boxes,
                                           std::vector<cv::Mat> query_boxes) {
  int N = boxes.size();
  int K = query_boxes.size();
  cv::Mat overlaps = cv::Mat::zeros(N, K, CV_64FC1);
  for (size_t k = 0; k < K; k++) {
    double box_area =
        (query_boxes[k].at<double>(0, 2) - query_boxes[k].at<double>(0, 0) +
         1) *
        (query_boxes[k].at<double>(0, 3) - query_boxes[k].at<double>(0, 1) + 1);
    for (size_t n = 0; n < N; n++) {
      double iw =
          std::min(boxes[n].at<double>(0, 2), query_boxes[k].at<double>(0, 2)) -
          std::max(boxes[n].at<double>(0, 0), query_boxes[k].at<double>(0, 0)) +
          1;
      if (iw > 0) {
        double ih = std::min(boxes[n].at<double>(0, 3),
                             query_boxes[k].at<double>(0, 3)) -
                    std::max(boxes[n].at<double>(0, 1),
                             query_boxes[k].at<double>(0, 1)) +
                    1;
        if (ih > 0) {
          double ua = static_cast<double>(
              (boxes[n].at<double>(0, 2) - boxes[n].at<double>(0, 0) + 1) *
                  (boxes[n].at<double>(0, 3) - boxes[n].at<double>(0, 1) + 1) +
              box_area - iw * ih);
          overlaps.at<double>(n, k) = iw * ih / ua;
        }
      }
    }
  }
  return overlaps;
}
cv::Mat FairmotMindsporePost::IouDistance(std::vector<STack *> atracks,
                                          std::vector<STack *> btracks) {
  std::vector<cv::Mat> atlbrs;
  std::vector<cv::Mat> btlbrs;
  cv::Mat cost_matrix;
  for (size_t i = 0; i < atracks.size(); i++) {
    atlbrs.push_back((*atracks[i]).tlbr());
  }
  for (size_t i = 0; i < btracks.size(); i++) {
    btlbrs.push_back((*btracks[i]).tlbr());
  }
  cv::Mat ious = cv::Mat::zeros(atlbrs.size(), btlbrs.size(), CV_64FC1);
  if (!ious.empty()) {
    ious = BboxOverlaps(atlbrs, btlbrs);
    cost_matrix = 1 - ious;
  } else {
    cost_matrix = cv::Mat::zeros(atlbrs.size(), btlbrs.size(), CV_64FC1);
  }
  return cost_matrix;
}
void FairmotMindsporePost::MultiPredict(std::vector<STack *> &stracks) {
  if (stracks.size() > 0) {
    cv::Mat multi_mean(stracks.size(), (*stracks[0]).mean.cols, CV_64FC1);
    std::vector<cv::Mat> multi_covariance;
    for (size_t i = 0; i < stracks.size(); i++) {
      multi_mean.row(i) = (*stracks[i]).mean.clone() + 0;
      multi_covariance.push_back((*stracks[i]).covariance);
    }
    for (size_t i = 0; i < stracks.size(); i++) {
      if ((*stracks[i]).state != TrackState_Tracked) {
        multi_mean.at<double>(i, 7) = 0;
      }
    }
    kalmanfilter.multi_predict(multi_mean, multi_covariance);
    for (size_t i = 0; i < multi_covariance.size(); i++) {
      (*stracks[i]).mean = multi_mean.row(i);
      (*stracks[i]).covariance = multi_covariance[i];
    }
  }
}

std::vector<STack *> FairmotMindsporePost::Get_output_stracks(
    MxBase::JDETracker &tracker, const std::vector<STack *> &activated_starcks,
    const std::vector<STack *> &refind_stracks,
    std::vector<STack *> lost_stracks, std::vector<STack *> removed_stracks) {
  std::vector<STack *> det_tmp;
  for (size_t i = 0; i < tracker.tracked_stracks.size(); i++) {
    if ((*tracker.tracked_stracks[i]).state == TrackState_Tracked) {
      det_tmp.push_back(tracker.tracked_stracks[i]);
    }
  }
  std::vector<STack *>().swap(tracker.tracked_stracks);
  tracker.tracked_stracks = det_tmp;
  std::vector<STack *>().swap(det_tmp);
  tracker.tracked_stracks =
      JointStracks(tracker.tracked_stracks, activated_starcks);
  tracker.tracked_stracks =
      JointStracks(tracker.tracked_stracks, refind_stracks);
  tracker.lost_stracks =
      SubStracks(tracker.lost_stracks, tracker.tracked_stracks);
  for (size_t i = 0; i < lost_stracks.size(); i++) {
    tracker.lost_stracks.push_back(lost_stracks[i]);
  }
  tracker.lost_stracks =
      SubStracks(tracker.lost_stracks, tracker.removed_stracks);
  for (size_t i = 0; i < removed_stracks.size(); i++) {
    tracker.removed_stracks.push_back(removed_stracks[i]);
  }
  std::vector<STack *> output_stracks;
  RemoveDuplicateStracks(tracker.tracked_stracks, tracker.lost_stracks);
  // get scores of lost tracks
  for (size_t i = 0; i < tracker.tracked_stracks.size(); i++) {
    if ((*tracker.tracked_stracks[i]).is_activated) {
      output_stracks.push_back(tracker.tracked_stracks[i]);
    }
  }
  return output_stracks;
}

void FairmotMindsporePost::Get_detections(cv::Mat det,
                                          std::vector<STack *> &detections,
                                          cv::Mat id_feature) {
  if (det.rows > 0) {
    cv::Mat det_tmp = det(cv::Range(0, det.rows), cv::Range(0, 5)).clone();
    for (size_t x = 0; x < det.rows; ++x) {
      cv::Mat tlbrs = det_tmp.row(x);
      cv::Mat f = id_feature.row(x);
      cv::Mat ret = tlbrs(cv::Range(0, tlbrs.rows), cv::Range(0, 4));
      for (size_t y = 0; y < 2; ++y) {
        ret.at<float>(0, y + 2) -= ret.at<float>(0, y);
      }
      STack *stack = new STack(ret, tlbrs.at<float>(0, 4), f, 30);
      detections.push_back(stack);
    }
  }
}

void FairmotMindsporePost::Get_dists(cv::Mat &dists,
                                     std::vector<STack *> &detections,
                                     std::vector<STack *> &strack_pool) {
  if (dists.rows != 0) {
    cv::Mat det_features(detections.size(), (*detections[0]).curr_feat.cols,
                         CV_32FC1);
    cv::Mat track_features(strack_pool.size(),
                           (*strack_pool[0]).smooth_feat.cols, CV_32FC1);
    for (size_t i = 0; i < detections.size(); i++)
      det_features.row(i) = (*detections[i]).curr_feat + 0;
    det_features.convertTo(det_features, CV_64F);
    for (size_t i = 0; i < strack_pool.size(); i++)
      track_features.row(i) = (*strack_pool[i]).smooth_feat + 0;
    track_features.convertTo(track_features, CV_64F);
    // cv::Mat cdist(track_features.rows, det_features.rows, CV_64FC1);
    for (size_t i = 0; i < dists.rows; i++)
      for (size_t j = 0; j < dists.cols; j++) {
        cv::normalize(det_features.row(j), det_features.row(j));
        cv::normalize(track_features.row(i), track_features.row(i));
        dists.at<double>(i, j) =
            1 - track_features.row(i).dot(det_features.row(j));
      }
  }
}

void FairmotMindsporePost::Update_Starcks(
    const std::vector<STack *> &strack_pool, std::vector<STack *> &detections,
    const cv::Mat &matches, std::vector<STack *> &activated_starcks,
    std::vector<STack *> &refind_stracks, const MxBase::JDETracker &tracker) {
  for (size_t i = 0; i < matches.rows; i++) {
    STack *track = strack_pool[matches.at<float>(i, 0)];
    STack *dets = detections[matches.at<float>(i, 1)];
    if ((*track).state == TrackState_Tracked) {
      (*track).update((*detections[matches.at<float>(i, 1)]), tracker.frame_id);
      activated_starcks.push_back(track);
    } else {
      (*track).re_activate(*dets, tracker.frame_id, false);
      refind_stracks.push_back(track);
    }
  }
}

std::vector<STack *> FairmotMindsporePost::ObjectDetectionOutput(
    const std::vector<TensorBase> &tensors, MxBase::JDETracker &tracker) {
  tracker.frame_id += 1;
  cv::Mat id_feature, det, matches, u_track, u_detection, u_unconfirmed, dists;
  std::vector<STack *> activated_starcks, refind_stracks, lost_stracks,
      removed_stracks, detections, unconfirmed, tracked_stracks, det_tmp,
      r_tracked_stracks, output_stracks, strack_pool;
  TensorBaseToCVMat(det, tensors[1]);
  TensorBaseToCVMat(id_feature, tensors[0]);
  PostProcess(det, tracker);
  cv::Mat scores = det(cv::Range(0, det.rows), cv::Range(4, 5));
  cv::Mat new_det(0, det.cols, CV_32FC1), new_id(0, id_feature.cols, CV_32FC1);
  for (size_t x = 0; x < det.rows; ++x) {
    if (det.at<float>(x, 4) > CONF_THRES) {
      new_det.push_back(det.row(x));
      new_id.push_back(id_feature.row(x));
    }
  }
  det = new_det;
  id_feature = new_id;
  Get_detections(det, detections, id_feature);
  for (size_t i = 0; i < tracker.tracked_stracks.size(); i++)
    if (!(*tracker.tracked_stracks[i]).is_activated)
      unconfirmed.push_back(tracker.tracked_stracks[i]);
    else
      tracked_stracks.push_back(tracker.tracked_stracks[i]);
  strack_pool = JointStracks(tracked_stracks, tracker.lost_stracks);
  MultiPredict(strack_pool);
  dists = cv::Mat::zeros(strack_pool.size(), detections.size(), CV_64FC1);
  Get_dists(dists, detections, strack_pool);
  FuseMotion(tracker.kalman_filter, dists, strack_pool, detections);
  std::vector<cv::Mat> results;
  results = LinearAssignment(dists, 0.4);
  matches = results[0];
  u_track = results[1];
  u_detection = results[2];
  Update_Starcks(strack_pool, detections, matches, activated_starcks,
                 refind_stracks, tracker);
  for (size_t i = 0; i < u_detection.cols; ++i)
    det_tmp.push_back(
        detections[static_cast<int>(u_detection.at<float>(0, i))]);
  detections = det_tmp;
  std::vector<STack *>().swap(det_tmp);
  for (size_t i = 0; i < u_track.cols; ++i)
    if ((*strack_pool[u_track.at<float>(0, i)]).state == TrackState_Tracked)
      r_tracked_stracks.push_back(strack_pool[u_track.at<float>(0, i)]);
  dists = IouDistance(r_tracked_stracks, detections);
  results = LinearAssignment(dists, 0.5);
  matches = results[0];
  u_track = results[1];
  u_detection = results[2];
  Update_Starcks(r_tracked_stracks, detections, matches, activated_starcks,
                 refind_stracks, tracker);
  for (size_t i = 0; i < u_track.cols; i++) {
    STack *track = r_tracked_stracks[u_track.at<float>(0, i)];
    if ((*track).state != TrackState_Lost) {
      (*track).state = TrackState_Lost;
      lost_stracks.push_back(track);
    }
  }
  for (size_t i = 0; i < u_detection.cols; ++i)
    det_tmp.push_back(
        detections[static_cast<int>(u_detection.at<float>(0, i))]);
  detections = det_tmp;
  std::vector<STack *>().swap(det_tmp);
  dists = IouDistance(unconfirmed, detections);
  results = LinearAssignment(dists, 0.7);
  matches = results[0];
  u_unconfirmed = results[1];
  u_detection = results[2];
  for (size_t i = 0; i < matches.rows; i++) {
    (*unconfirmed[matches.at<float>(i, 0)])
        .update((*detections[matches.at<float>(i, 1)]), tracker.frame_id);
    activated_starcks.push_back(unconfirmed[matches.at<float>(i, 0)]);
  }
  for (size_t i = 0; i < u_unconfirmed.cols; i++) {
    STack *track = unconfirmed[u_unconfirmed.at<float>(0, i)];
    (*track).state = TrackState_Removed;
    removed_stracks.push_back(track);
  }
  for (int j = 0; j < u_detection.cols; j++) {
    auto inew = u_detection.at<float>(0, j);
    STack *track = detections[inew];
    if ((*track).score < tracker.det_thresh) continue;
    (*track).activate(tracker.kalman_filter, tracker.frame_id);
    activated_starcks.push_back(track);
  }
  for (size_t i = 0; i < tracker.lost_stracks.size(); i++) {
    if (tracker.frame_id - (*tracker.lost_stracks[i]).frame_id >
        tracker.max_time_lost) {
      (*tracker.lost_stracks[i]).state = TrackState_Removed;
      removed_stracks.push_back(tracker.lost_stracks[i]);
    }
  }
  output_stracks =
      Get_output_stracks(tracker, activated_starcks, refind_stracks,
                         lost_stracks, removed_stracks);
  return output_stracks;
}

APP_ERROR FairmotMindsporePost::Process(const std::vector<TensorBase> &tensors,
                                        MxBase::JDETracker &tracker,
                                        MxBase::Files &file) {
  LogDebug << "Begin to process FairmotMindsporePost.";
  auto inputs = tensors;
  APP_ERROR ret = CheckAndMoveTensors(inputs);
  if (ret != APP_ERR_OK) {
    LogError << "CheckAndMoveTensors failed, ret=" << ret;
    return ret;
  }
  std::vector<STack *> online_targets;
  online_targets = ObjectDetectionOutput(inputs, tracker);
  std::vector<cv::Mat> online_tlwhs;
  std::vector<int> online_ids;
  for (size_t i = 0; i < online_targets.size(); i++) {
    cv::Mat tlwh = (*online_targets[i]).gettlwh();
    int tid = (*online_targets[i]).track_id;
    double tmp = tlwh.at<double>(0, 2) / tlwh.at<double>(0, 3);
    bool vertical = false;
    if (tmp > 1.6) {
      vertical = true;
    }
    if ((tlwh.at<double>(0, 2) * tlwh.at<double>(0, 3) > MIN_BOX_AREA) &&
        vertical == false) {
      online_tlwhs.push_back(tlwh);
      online_ids.push_back(tid);
    }
  }
  Results *result = new Results(file.frame_id + 1, online_tlwhs, online_ids);
  file.results.push_back(result);
  file.frame_id += 1;
  LogInfo << "End to process FairmotMindsporePost.";
  return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::FairmotMindsporePost> GetObjectInstance() {
  LogInfo << "Begin to get FairmotMindsporePost instance.";
  auto instance = std::make_shared<FairmotMindsporePost>();
  LogInfo << "End to get FairmotMindsporePost Instance";
  return instance;
}
}

}  // namespace MxBase
