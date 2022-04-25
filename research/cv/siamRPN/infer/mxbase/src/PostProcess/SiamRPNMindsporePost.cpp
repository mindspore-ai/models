/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "SiamRPNMindsporePost.h"

#include <geos_c.h>

#include <memory>
#include <string>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "acl/acl.h"

namespace MxBase {
const int OUTPUT_TENSOR_SIZE = 2;
const int OUTPUT_1 = 0;
const int OUTPUT_2 = 1;
const int OUTPUT_1_SIZE = 3;
const int OUTPUT_2_SIZE = 3;

SiamRPNMindsporePost &SiamRPNMindsporePost::operator=(
    const SiamRPNMindsporePost &other) {
  if (this == &other) {
    return *this;
  }
  ObjectPostProcessBase::operator=(other);
  return *this;
}
bool SiamRPNMindsporePost::IsValidTensors(
    const std::vector<TensorBase> &tensors) const {
  if (tensors.size() < OUTPUT_TENSOR_SIZE) {
    LogError << "The number of tensor (" << tensors.size()
             << ") is less than required (" << OUTPUT_TENSOR_SIZE << ")";
    return false;
  }

  auto idFeatureShape = tensors[OUTPUT_1].GetShape();
  if (idFeatureShape.size() != OUTPUT_1_SIZE) {
    LogError << "The number of tensor[" << OUTPUT_1
             << "] dimensions (" << idFeatureShape.size()
             << ") is not equal to (" << OUTPUT_1_SIZE << ")";
    return false;
  }

  auto detsShape = tensors[OUTPUT_2].GetShape();
  if (detsShape.size() != OUTPUT_2_SIZE) {
    LogError << "The number of tensor[" << OUTPUT_2 << "] dimensions ("
             << detsShape.size() << ") is not equal to ("
             << OUTPUT_2_SIZE << ")";
    return false;
  }
  return true;
}

void SiamRPNMindsporePost::TensorBaseToCVMat(cv::Mat &imageMat,
                                             const MxBase::TensorBase &tensor) {
  TensorBase Data = tensor;
  uint32_t outputModelWidth;
  uint32_t outputModelHeight;
  auto shape = Data.GetShape();
  if (shape.size() == 2) {
    outputModelHeight = shape[0];
    outputModelWidth = shape[1];
  } else {
    outputModelHeight = shape[1];
    outputModelWidth = shape[2];
  }
  auto *data = reinterpret_cast<float *>(GetBuffer(Data, 0));
  cv::Mat dataMat(outputModelHeight, outputModelWidth, CV_32FC1);
  for (size_t y = 0; y < outputModelHeight; ++y) {
    for (size_t x = 0; x < outputModelWidth; ++x) {
      dataMat.at<float>(y, x) = data[y * outputModelWidth + x];
    }
  }
  imageMat = dataMat.clone();
}

int softmax(const cv::Mat& src, cv::Mat& dst) {
  cv::Mat col1 = src.colRange(0, 1).clone();
  cv::Mat col2 = src.colRange(1, 2).clone();
  cv::exp(col1, col1);
  cv::exp(col2, col2);
  cv::add(col1, col2, col1);
  cv::divide(col2, col1, dst);
  return 0;
}

std::vector<cv::Mat> box_transform_inv(const cv::Mat &src,
                                       cv::Mat &offset) {
  cv::Mat anchor_xctr = src.colRange(0, 1).clone();
  cv::Mat anchor_yctr = src.colRange(1, 2).clone();
  cv::Mat anchor_w = src.colRange(2, 3).clone();
  cv::Mat anchor_h = src.colRange(3, 4).clone();
  cv::Mat offset_x = offset.colRange(0, 1).clone();
  cv::Mat offset_y = offset.colRange(1, 2).clone();
  cv::Mat offset_w = offset.colRange(2, 3).clone();
  cv::Mat offset_h = offset.colRange(3, 4).clone();

  cv::Mat box_cx, box_cy, box_w, box_h;
  cv::multiply(anchor_w, offset_x, box_cx);
  box_cx = box_cx + anchor_xctr;
  cv::multiply(anchor_h, offset_y, box_cy);
  box_cy = box_cy + anchor_yctr;
  cv::exp((offset_w), offset_w);
  cv::multiply(anchor_w, offset_w, box_w);
  cv::exp((offset_h), offset_h);
  cv::multiply(anchor_h, offset_h, box_h);
  std::vector<cv::Mat> channels;
  channels.push_back(box_cx);
  channels.push_back(box_cy);
  channels.push_back(box_w);
  channels.push_back(box_h);
  return channels;
}
cv::Mat readMatFromFile(std::string path, int height, int width) {
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  cv::Mat im(height, width, CV_32FC1);
  if (!inFile) {
    std::cout << "error" << std::endl;
    return im;
  }

  for (int r = 0; r < im.rows; r++) {
    inFile.read(reinterpret_cast<char *>(im.ptr<uchar>(r)),
                im.cols * im.elemSize());
  }
  inFile.close();
  return im;
}

cv::Mat sz(std::vector<cv::Mat> &bbox) {
  cv::Mat w = bbox[2].clone();
  cv::Mat h = bbox[3].clone();
  cv::Mat pad = (w + h) * 0.5;
  cv::Mat sz2, temp;
  cv::multiply((w + pad), (h + pad), temp);
  cv::sqrt(temp, sz2);
  return sz2;
}

float sz_wh(float *wh, float scale) {
  float wh1 = wh[0] * scale;
  float wh2 = wh[1] * scale;
  float pad = (wh1 + wh2) * 0.5;
  float sz2 = (wh1 + pad) * (wh2 + pad);
  return sqrt(sz2);
}

cv::Mat change(cv::Mat &r) { return cv::max(r, 1 / r); }

cv::Mat get_rc(const std::vector<cv::Mat> &box_pred, float *wh, float scale) {
  float ratio = wh[0] / wh[1];
  cv::Mat temp1 = box_pred[2] / box_pred[3];
  temp1 = temp1 / ratio;
  temp1 = cv::max(temp1, 1 / temp1);
  return temp1;
}

cv::Mat get_sc(std::vector<cv::Mat> &box_pred, float *wh, float scale) {
  cv::Mat temp1 = sz(box_pred);
  float ss = sz_wh(wh, scale);
  temp1 = temp1 / ss;
  temp1 = cv::max(temp1, 1 / temp1);
  return temp1;
}

cv::Mat get_penalty(const cv::Mat &s_c, cv::Mat &r_c, float penalty_k) {
  cv::Mat mm;
  cv::multiply(s_c, r_c, mm);
  mm = -(mm - 1) * penalty_k;
  cv::exp((mm), mm);
  return mm;
}

static void geos_message_handler(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
}

bool judge_failures(float *pred_box, float *gt_box) {
  initGEOS(geos_message_handler, geos_message_handler);
  std::string a =
      std::to_string(pred_box[0]) + ' ' + std::to_string(pred_box[1]) + ", " +
      std::to_string(pred_box[0]) + ' ' + std::to_string(pred_box[3]) + ", " +
      std::to_string(pred_box[2]) + ' ' + std::to_string(pred_box[3]) + ", " +
      std::to_string(pred_box[2]) + ' ' + std::to_string(pred_box[1]) + ", " +
      std::to_string(pred_box[0]) + ' ' + std::to_string(pred_box[1]);
  std::string b =
      std::to_string(gt_box[6]) + ' ' + std::to_string(gt_box[7]) + ", " +
      std::to_string(gt_box[0]) + ' ' + std::to_string(gt_box[1]) + ", " +
      std::to_string(gt_box[2]) + ' ' + std::to_string(gt_box[3]) + ", " +
      std::to_string(gt_box[4]) + ' ' + std::to_string(gt_box[5]) + ", " +
      std::to_string(gt_box[6]) + ' ' + std::to_string(gt_box[7]);
  std::string wkt_a = "POLYGON((" + a + "))";
  std::string wkt_b = "POLYGON((" + b + "))";
  std::cout << wkt_a << " " << wkt_b << std::endl;
  /* Read the WKT into geometry objects */
  GEOSWKTReader *reader = GEOSWKTReader_create();
  GEOSGeometry *geom_a = GEOSWKTReader_read(reader, wkt_a.c_str());
  GEOSGeometry *geom_b = GEOSWKTReader_read(reader, wkt_b.c_str());
  /* Calculate the intersection */
  GEOSWKTWriter *writer = GEOSWKTWriter_create();
  GEOSGeometry *inter = GEOSIntersection(geom_a, geom_b);
  GEOSWKTWriter_setTrim(writer, 1);
  std::string wkt_inter = GEOSWKTWriter_write(writer, inter);
  std::cout << wkt_inter << std::endl;
  if (wkt_inter == "GEOMETRYCOLLECTION EMPTY" || wkt_inter == "POLYGON EMPTY") {
    return false;
  } else {
    return true;
  }
}
int read_gtBox(std::string path, float gt_bbox[][8]) {
  std::ifstream infile(path);
  std::cout << path << std::endl;
  int k = 0;
  char s;
  while (infile >> gt_bbox[k][0] >> s >> gt_bbox[k][1] >> s >> gt_bbox[k][2] >>
         s >> gt_bbox[k][3] >> s >> gt_bbox[k][4] >> s >> gt_bbox[k][5] >> s >>
         gt_bbox[k][6] >> s >> gt_bbox[k][7]) {
    k++;
  }
  return k;
}
void copy_box(float *box, float *bbox, int num) {
  for (int i = 0; i < num; i++) {
    box[i] = bbox[i];
  }
}
float getrange(float num, float min, float max) {
  float temp = num;
  if (num > max) {
    temp = max;
  } else if (num < min) {
    temp = min;
  }
  return temp;
}

APP_ERROR SiamRPNMindsporePost::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
  LogInfo << "Begin to initialize SiamRPNMindsporePost.";
  APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Fail to superinit  in ObjectPostProcessBase.";
    return ret;
  }
  LogInfo << "End to initialize SiamRPNMindsporePost.";
  return APP_ERR_OK;
}

APP_ERROR SiamRPNMindsporePost::DeInit() {
  LogInfo << "Begin to deinitialize SiamRPNMindsporePost.";
  LogInfo << "End to deinitialize SiamRPNMindsporePost.";
  return APP_ERR_OK;
}

void SiamRPNMindsporePost::ObjectTrackingOutput(
    const std::vector<TensorBase> &tensors, Config* postConfig,
    Tracker &track, int idx, int total_num, float result_box[][4],
    int &template_idx) {
  LogDebug << "SiamRPNMindsporePost start to write results.";
  cv::Mat rout, ccout, im, im_1;
  double maxValue, minValue;
  int minId, maxId;
  float cx, cy, w, h;
  float terget[4];
  TensorBaseToCVMat(ccout, tensors[0]);
  TensorBaseToCVMat(rout, tensors[1]);
  // sofMax
  cv::Mat ccout_sofmax;
  softmax(ccout, ccout_sofmax);
  std::vector<cv::Mat> box_pred = box_transform_inv(postConfig->anchors, rout);
  cv::Mat s_c = get_sc(box_pred, track.target_sz, track.scale_x);
  cv::Mat r_c = get_rc(box_pred, track.target_sz, track.scale_x);
  cv::Mat penalty = get_penalty(s_c, r_c, postConfig->penalty_k);
  cv::Mat score_pred = ccout_sofmax.colRange(0, 1).clone();
  cv::Mat pscore;
  cv::multiply(score_pred, penalty, pscore);
  pscore = pscore * (1 - postConfig->window_influence) +
           postConfig->windows * postConfig->window_influence;
  cv::minMaxIdx(pscore, &minValue, &maxValue, &minId, &maxId);
  cx = box_pred[0].at<float>(maxId, 0);
  cy = box_pred[1].at<float>(maxId, 0);
  w = box_pred[2].at<float>(maxId, 0);
  h = box_pred[3].at<float>(maxId, 0);
  std::cout << "transfer box:" << cx << " " << cy << " " << w << " " << h << " "
            << maxId << std::endl;
  terget[0] = cx / track.scale_x;
  terget[1] = cy / track.scale_x;
  terget[2] = w / track.scale_x;
  terget[3] = h / track.scale_x;
  float lr = penalty.at<float>(maxId, 0) * score_pred.at<float>(maxId, 0) *
             postConfig->lr_box;
  std::cout << "terget" << terget[0] << " " << terget[1] << " " << terget[2]
            << " " << terget[3] << " " << maxId << std::endl;
  std::cout << "target_sz lr" << track.pos[0] << " " << track.pos[1] << " "
            << track.target_sz[0] << " " << track.target_sz[1] << " " << lr
            << std::endl;
  float res_x = getrange(terget[0] + track.pos[0], 0, track.shape[1]);
  float res_y = getrange(terget[1] + track.pos[1], 0, track.shape[0]);
  float res_w = getrange(track.target_sz[0] * (1 - lr) + terget[2] * lr,
                         postConfig->min_scale * track.origin_target_sz[0],
                         postConfig->max_scale * track.origin_target_sz[0]);
  float res_h = getrange(track.target_sz[1] * (1 - lr) + terget[3] * lr,
                         postConfig->min_scale * track.origin_target_sz[1],
                         postConfig->max_scale * track.origin_target_sz[1]);
  track.pos[0] = res_x;
  track.pos[1] = res_y;
  track.target_sz[0] = res_w;
  track.target_sz[1] = res_h;
  track.pred_box[0] = getrange(res_x, 0, track.shape[1]);
  track.pred_box[1] = getrange(res_y, 0, track.shape[0]);
  track.pred_box[2] = getrange(res_w, 10, track.shape[1]);
  track.pred_box[3] = getrange(res_h, 10, track.shape[0]);
  float resultbox[4];
  float tempx = track.pred_box[0];
  float tempy = track.pred_box[1];
  resultbox[0] = track.pred_box[0] - track.pred_box[2] / 2 + 0.5;
  resultbox[1] = track.pred_box[1] - track.pred_box[3] / 2 + 0.5;
  resultbox[2] = tempx + track.pred_box[2] / 2 - 0.5;
  resultbox[3] = tempy + track.pred_box[3] / 2 - 0.5;
  track.pred_box[0] = resultbox[0];
  track.pred_box[1] = resultbox[1];
  bool flag = judge_failures(resultbox, track.gt_bbox[idx]);
  if (flag == 1) {
    copy_box(result_box[idx], resultbox, 4);
    copy_box(track.box_01, track.pred_box, 4);
  } else {
    result_box[idx][0] = 2;
    result_box[idx][1] = 0;
    result_box[idx][2] = 0;
    result_box[idx][3] = 0;
    template_idx = std::min(idx + 5, total_num - 1);
  }

  LogDebug << "SiamRPNMindsporePost write results succeeded.";
}

APP_ERROR SiamRPNMindsporePost::Process(const std::vector<TensorBase> &tensors,
                                        MxBase::Config &postConfig,
                                        MxBase::Tracker &track, int idx,
                                        int total_num, float result_box[][4],
                                        int &template_idx) {
  LogDebug << "Begin to process SiamRPNMindsporePost.";
  auto inputs = tensors;
  APP_ERROR ret = CheckAndMoveTensors(inputs);
  if (ret != APP_ERR_OK) {
    LogError << "CheckAndMoveTensors failed, ret=" << ret;
    return ret;
  }
  ObjectTrackingOutput(inputs, &postConfig, track, idx, total_num, result_box,
                       template_idx);
  LogInfo << "End to process SiamRPNMindsporePost.";
  return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::SiamRPNMindsporePost> GetObjectInstance() {
  LogInfo << "Begin to get SiamRPNMindsporePost instance.";
  auto instance = std::make_shared<SiamRPNMindsporePost>();
  LogInfo << "End to get SiamRPNMindsporePost Instance";
  return instance;
}
}

}  // namespace MxBase
