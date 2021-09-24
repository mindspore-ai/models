/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dirent.h>
#include <geos_c.h>
#include <gflags/gflags.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "../inc/utils.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/vision.h"
#include "include/dataset/transforms.h"
#include "include/dataset/execute.h"


namespace ms = mindspore;
DEFINE_string(siamRPN_file, "", "mindir path");
DEFINE_string(image_path, "", "dataset path");
DEFINE_string(dataset_name, "", "dataset name");
DEFINE_int32(device_id, 0, "device id");

float min_box(float* bbox, int start, int step, int len) {
  float min_value = bbox[start];
  for (int i = start; i < len; i = i + step) {
    if (min_value > bbox[i]) {
      min_value = bbox[i];
    }
  }
  return min_value;
}

float max(float* bbox, int start, int step, int len) {
  float max_value = bbox[start];
  for (int i = start; i < len; i = i + step) {
    if (max_value < bbox[i]) {
      max_value = bbox[i];
    }
  }
  return max_value;
}
void trans_box(float* bbox, float* box) {
  float x1 = min_box(bbox, 0, 2, 8);
  float x2 = max(bbox, 0, 2, 8);
  float y1 = min_box(bbox, 1, 2, 8);
  float y2 = max(bbox, 1, 2, 8);
  float distance_1 = bbox[0] - bbox[2];
  float distance_2 = bbox[1] - bbox[3];
  float distance_3 = bbox[2] - bbox[4];
  float distance_4 = bbox[3] - bbox[5];
  float w = std::sqrt(distance_1 * distance_1 + distance_2 * distance_2);
  float h = std::sqrt(distance_3 * distance_3 + distance_4 * distance_4);
  float A1 = w * h;
  float A2 = (x2 - x1) * (y2 - y1);
  float s = std::sqrt(A1 / A2);
  w = s * (x2 - x1) + 1;
  h = s * (y2 - y1) + 1;
  float x = x1;
  float y = y1;
  box[0] = x;
  box[1] = y;
  box[2] = w;
  box[3] = h;
}


cv::Mat Pad(const cv::Mat& srcImageMat, int left, int bottom,
         int right, int top) {
  cv::Mat dstImageMat;
  cv::Scalar tempVal = cv::mean(srcImageMat);
  tempVal.val[0] = static_cast<int>(tempVal.val[0]);
  tempVal.val[1] = static_cast<int>(tempVal.val[1]);
  tempVal.val[2] = static_cast<int>(tempVal.val[2]);
  int borderType = cv::BORDER_CONSTANT;
  copyMakeBorder(srcImageMat, dstImageMat, top, bottom, left, right, borderType,
                 tempVal);
  return dstImageMat;
}
// area is the upper left corner coordinates and width and height of the cutting area
cv::Mat Crop(const cv::Mat& img, const std::vector<int>& area) {
  cv::Mat crop_img;
  int crop_x1 = std::max(0, area[0]);
  int crop_y1 = std::max(0, area[1]);
  int crop_x2 = std::min(img.cols - 1, area[0] + area[2] - 1);
  int crop_y2 = std::min(img.rows - 1, area[1] + area[3] - 1);
  crop_img = img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
  return crop_img;
}
cv::Mat ResizeImage(const cv::Mat& srcImageMat, const std::vector<int>& size) {
  cv::Mat dstImageMat;
  cv::resize(srcImageMat, dstImageMat, cv::Size(size[0], size[1]));
  return dstImageMat;
}
cv::Mat get_template_Mat(const std::string &file_path, float* box, int resize_template,
                         float context_amount) {
  cv::Mat srcImageMat;
  srcImageMat = cv::imread(file_path, cv::IMREAD_COLOR);
  int w = srcImageMat.cols;
  int h = srcImageMat.rows;
  int cx = box[0] + box[2] / 2 - 1 / 2;
  int cy = box[1] + box[3] / 2 - 1 / 2;
  float w_template = box[2] + (box[2] + box[3]) * context_amount;
  float h_template = box[3] + (box[2] + box[3]) * context_amount;
  float s_x = std::sqrt(w_template * h_template);

  int left_x = cx - (s_x - 1) / 2 + w;
  int top_y = cy - (s_x - 1) / 2 + h;
  std::vector<int> position = {left_x, top_y, static_cast<int>(s_x), static_cast<int>(s_x)};
  std::vector<int> size = {resize_template, resize_template};
  srcImageMat = Pad(srcImageMat,  w, h, w, h);
  srcImageMat = Crop(srcImageMat, position);
  srcImageMat = ResizeImage(srcImageMat, size);
  // HWC2CHW
  cv::Mat srcImageMat1;
  srcImageMat.convertTo(srcImageMat1, CV_32FC3);
  std::vector<float> dst_data;
  std::vector<cv::Mat> bgrChannels(3);

  cv::split(srcImageMat1, bgrChannels);
  for (auto i = 0; i < bgrChannels.size(); i++) {
    std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
    dst_data.insert(dst_data.end(), data.begin(), data.end());
  }
  srcImageMat1 = cv::Mat(dst_data, true);
  cv::Mat dst = srcImageMat1.reshape(3, 127);
  return dst;
}

cv::Mat get_detection_Mat(std::string file_path, float* box,
                          int resize_template, int resize_detection,
                          float context_amount, float* scale_x) {
  cv::Mat srcImageMat;
  srcImageMat = cv::imread(file_path, cv::IMREAD_COLOR);
  int w = srcImageMat.cols;
  int h = srcImageMat.rows;
  int cx = box[0] + box[2] / 2 - 1 / 2;
  int cy = box[1] + box[3] / 2 - 1 / 2;
  float w_template = box[2] + (box[2] + box[3]) * context_amount;
  float h_template = box[3] + (box[2] + box[3]) * context_amount;
  float s_x = std::sqrt(w_template * h_template);
  s_x = s_x * resize_detection / resize_template;
  *scale_x = resize_detection /  static_cast<float>(s_x);
  s_x = static_cast<int>(s_x);
  int left_x = cx - (s_x - 1) / 2 + w;
  int top_y = cy - (s_x - 1) / 2 + h;

  std::vector<int> position = {left_x, top_y, static_cast<int>(s_x), static_cast<int>(s_x)};
  std::vector<int> size = {resize_detection, resize_detection};

  srcImageMat = Pad(srcImageMat,  w, h, w, h);
  srcImageMat = Crop(srcImageMat, position);
  srcImageMat = ResizeImage(srcImageMat, size);


  // HWC2CHW
  cv::Mat srcImageMat1;
  srcImageMat.convertTo(srcImageMat1, CV_32FC3);
  std::vector<float> dst_data;
  std::vector<cv::Mat> bgrChannels(3);

  cv::split(srcImageMat1, bgrChannels);

  for (auto i = 0; i < bgrChannels.size(); i++) {
    std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
    dst_data.insert(dst_data.end(), data.begin(), data.end());
  }
  srcImageMat1 = cv::Mat(dst_data, true);
  cv::Mat dst = srcImageMat1.reshape(3, resize_detection);
  return dst;
}

// postprocess using
float getrange(float num, float min, float max) {
  float temp = num;
  if (num > max) {
    temp = max;
  } else if (num < min) {
    temp = min;
  }
  return temp;
}

cv::Mat softmax(const cv::Mat& src) {
  cv::Mat dst;
  cv::Mat col1 = src.colRange(0, 1).clone();
  cv::Mat col2 = src.colRange(1, 2).clone();
  cv::exp(col1, col1);
  cv::exp(col2, col2);
  cv::add(col1, col2, col1);
  cv::divide(col2, col1, dst);
  return dst;
}

std::vector<cv::Mat> box_transform_inv(const cv::Mat& src, const cv::Mat& offset) {
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
    inFile.read(reinterpret_cast<char*>(im.ptr<uchar>(r)), im.cols * im.elemSize());
  }
  inFile.close();
  return im;
}

cv::Mat sz(const std::vector<cv::Mat>& bbox) {
  cv::Mat w = bbox[2].clone();
  cv::Mat h = bbox[3].clone();
  cv::Mat pad = (w + h) * 0.5;
  cv::Mat sz2, temp;
  cv::multiply((w + pad), (h + pad), temp);
  cv::sqrt(temp, sz2);
  return sz2;
}

float sz_wh(float* wh, float scale) {
  float wh1 = wh[0] * scale;
  float wh2 = wh[1] * scale;
  float pad = (wh1 + wh2) * 0.5;
  float sz2 = (wh1 + pad) * (wh2 + pad);
  return sqrt(sz2);
}

cv::Mat change(const cv::Mat& r) { return cv::max(r, 1 / r); }

cv::Mat get_rc(const std::vector<cv::Mat>& box_pred, const float* wh, const float scale) {
  float ratio = wh[0] / wh[1];
  cv::Mat temp1 = box_pred[2] / box_pred[3];
  temp1 = temp1 / ratio;
  temp1 = cv::max(temp1, 1 / temp1);
  return temp1;
}

cv::Mat get_sc(const std::vector<cv::Mat>& box_pred, float* wh, float scale) {
  cv::Mat temp1 = sz(box_pred);
  float ss = sz_wh(wh, scale);
  temp1 = temp1 / ss;
  temp1 = cv::max(temp1, 1 / temp1);
  return temp1;
}

cv::Mat get_penalty(const cv::Mat& s_c, const cv::Mat& r_c, float penalty_k) {
  cv::Mat mm;
  cv::multiply(s_c, r_c, mm);
  mm = -(mm - 1) * penalty_k;
  cv::exp((mm), mm);
  return mm;
}

static void geos_message_handler(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
}

bool judge_failures(float* pred_box, float* gt_box) {
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
  /* Read the WKT into geometry objects */
  GEOSWKTReader* reader = GEOSWKTReader_create();
  GEOSGeometry* geom_a = GEOSWKTReader_read(reader, wkt_a.c_str());
  GEOSGeometry* geom_b = GEOSWKTReader_read(reader, wkt_b.c_str());

  /* Calculate the intersection */
  GEOSWKTWriter* writer = GEOSWKTWriter_create();
  GEOSGeometry* inter = GEOSIntersection(geom_a, geom_b);
  GEOSWKTWriter_setTrim(writer, 1);
  std::string wkt_inter = GEOSWKTWriter_write(writer, inter);

  if (wkt_inter == "POLYGON EMPTY") {
    return false;
  } else {
    return true;
  }
}
int read_gtBox(std::string path, float gt_bbox[][8]) {
  std::ifstream infile(path);
  int k = 0;
  char s;
  while (infile >> gt_bbox[k][0] >> s >> gt_bbox[k][1] >> s >> gt_bbox[k][2] >>
         s >> gt_bbox[k][3] >> s >> gt_bbox[k][4] >> s >> gt_bbox[k][5] >> s >>
         gt_bbox[k][6] >> s >> gt_bbox[k][7]) {
    k++;
  }
  return k;
}
void copy_box(float* box, float* bbox, int num) {
  for (int i = 0; i < num; i++) {
    box[i] = bbox[i];
  }
}
void copy_box_four_value(float* box, float value1, float value2, float value3, float value4) {
  box[0] = value1;
  box[1] = value2;
  box[2] = value3;
  box[3] = value4;
}
void copy_box_two_value(float* box, float value1, float value2) {
  box[0] = value1;
  box[1] = value2;
}
struct Config {
  int resize_template = 127;
  int resize_detection = 255;
  cv::Mat anchors = readMatFromFile("../ascend_310_infer/src/anchors.bin", 1445, 4);
  cv::Mat windows = readMatFromFile("../ascend_310_infer/src/windows.bin", 1445, 1);
  float context_amount = 0.5;
  float min_scale = 0.1;
  float max_scale = 10;
  float window_influence = 0.40;
  float penalty_k = 0.22;
  float lr_box = 0.3;
  float gt_bbox[2500][8];
  float pred_box[4];
  float scale_x = 0.0;
  float bbox[8];
  float box_01[4];
  bool flag = true;
  float target_sz[2];
  float pos[2];
  int infer_cout_shape[2]={1445, 2};
  int infer_rout_shape[2]={1445, 4};
};

void deal_predict(const cv::Mat ccout, const cv::Mat rout, struct Config* config, float *shape, float *origin_target_sz,
                  int i, float* resultbox) {
  float  cx, cy, w, h, terget[4];
  double maxValue, minValue;
  int minId, maxId = 0;
  cv::Mat ccout_sofmax, pscore;
  ccout_sofmax = softmax(ccout);
  std::vector<cv::Mat> box_pred = box_transform_inv(config->anchors, rout);
  cv::Mat s_c = get_sc(box_pred, config->target_sz, config->scale_x);
  cv::Mat r_c = get_rc(box_pred, config->target_sz, config->scale_x);
  cv::Mat penalty = get_penalty(s_c, r_c, config->penalty_k);
  cv::Mat score_pred = ccout_sofmax.colRange(0, 1).clone();
  cv::multiply(score_pred, penalty, pscore);
  pscore = pscore * (1 - config->window_influence) + config->windows * config->window_influence;
  cv::minMaxIdx(pscore, &minValue, &maxValue, &minId, &maxId);
  cx = box_pred[0].at<float>(maxId, 0);
  cy = box_pred[1].at<float>(maxId, 0);
  w = box_pred[2].at<float>(maxId, 0);
  h = box_pred[3].at<float>(maxId, 0);
  copy_box_four_value(terget, cx / config->scale_x, cy / config->scale_x, w / config->scale_x, h / config->scale_x);
  float lr = penalty.at<float>(maxId, 0) * score_pred.at<float>(maxId, 0) * config->lr_box;
  float res_x = getrange(terget[0] + config->pos[0], 0, shape[1]);
  float res_y = getrange(terget[1] + config->pos[1], 0, shape[0]);
  float res_w = getrange(config->target_sz[0] * (1 - lr) + terget[2] * lr,
                         config->min_scale * origin_target_sz[0], config->max_scale * origin_target_sz[0]);
  float res_h = getrange(config->target_sz[1] * (1 - lr) + terget[3] * lr,
                         config->min_scale * origin_target_sz[1], config->max_scale * origin_target_sz[1]);
  copy_box_two_value(config->pos, res_x, res_y);
  copy_box_two_value(config->target_sz, res_w, res_h);
  copy_box_four_value(config->pred_box, getrange(res_x, 0, shape[1]), getrange(res_y, 0, shape[0]),
                      getrange(res_w, 0, shape[1]), getrange(res_h, 0, shape[0]));
  resultbox[0] = config->pred_box[0] - config->pred_box[2] / 2 + 1 / 2;
  resultbox[1] = config->pred_box[1] - config->pred_box[3] / 2 + 1 / 2;
  resultbox[2] = config->pred_box[0] + config->pred_box[2] / 2 - 1 / 2;
  resultbox[3] = config->pred_box[1] + config->pred_box[3] / 2 - 1 / 2;
  copy_box_two_value(config->pred_box, resultbox[0], resultbox[1]);
  config->flag = judge_failures(resultbox, config->gt_bbox[i]);
}
int process_infer(const std::vector<std::string>& dirs, const std::vector<ms::MSTensor>& model_inputs,
             const std::string& data_set, ms::Model* siamRPN, ms::Status ret, std::map<double, double>* costTime_map) {
  Config config;
  for (const auto &dir : dirs) {
    std::vector<std::string> images = GetAllFiles(data_set + '/' + FLAGS_dataset_name + '/' + dir + "/color");
    int k = read_gtBox(data_set + '/' + FLAGS_dataset_name + '/' + dir + "/groundtruth.txt", config.gt_bbox);
    int template_idx = 0;
    float result_box[k][4], resultbox[4], shape[2], origin_target_sz[2];
    std::vector<ms::MSTensor> inputs;
    std::string image_template, image_detection;
    for (int i = 0; i < static_cast<int>(images.size()); i++) {
      struct timeval start;
      struct timeval end;
      double startTime_ms;
      double endTime_ms;
      std::cout << "start infer：" << i << " " << template_idx << " " << std::endl;
      if (i == template_idx) {
        cv::Mat imageMat = cv::imread(images[i], cv::IMREAD_COLOR);
        shape[0] = imageMat.rows;
        shape[1] = imageMat.cols;
        copy_box(config.bbox, config.gt_bbox[i], 8);
        trans_box(config.bbox, config.box_01);
        config.pos[0] = config.box_01[0] + (config.box_01[2] + 1) / 2;
        config.pos[1] = config.box_01[1] + (config.box_01[3] + 1) / 2;
        copy_box_two_value(config.target_sz, config.box_01[2], config.box_01[3]);
        copy_box_two_value(origin_target_sz, config.box_01[2], config.box_01[3]);
        image_template = images[template_idx];
        cv::Mat srcImageMat = get_template_Mat(image_template, config.box_01, config.resize_template,
        config.context_amount);
        size_t size_buffer = srcImageMat.size().height * srcImageMat.size().width * 3 * 4;
        mindspore::MSTensor buffer("template", mindspore::DataType::kNumberTypeFloat32,
            {static_cast<int64_t>(3), static_cast<int64_t>(srcImageMat.size().height),
            static_cast<int64_t>(srcImageMat.size().width)}, srcImageMat.data, size_buffer);
        inputs.clear();
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(),
        model_inputs[0].Shape(), buffer.Data().get(), buffer.DataSize());
        copy_box_four_value(result_box[i], 1.0, 0.0, 0.0, 0.0);
      } else if (i < template_idx) {
        copy_box_four_value(result_box[i], 0.0, 0.0, 0.0, 0.0);
      } else {
        std::vector<ms::MSTensor> outputs;
        image_detection = images[i];
        cv::Mat decImageMat = get_detection_Mat(image_detection, config.box_01, config.resize_template,
                              config.resize_detection, config.context_amount, &config.scale_x);
        size_t size_detection_buffer = decImageMat.size().height * decImageMat.size().width * 3 * 4;
        auto dec_size = decImageMat.size();
        mindspore::MSTensor buffer1("detection", mindspore::DataType::kNumberTypeFloat32, {static_cast<int64_t>(3),
                                    static_cast<int64_t>(decImageMat.size().height),
                                    static_cast<int64_t>(decImageMat.size().width)},
                                    decImageMat.data, size_detection_buffer);
        inputs.emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(),
                            model_inputs[1].Shape(), buffer1.Data().get(), buffer1.DataSize());
        gettimeofday(&start, NULL);
        ret = siamRPN->Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != ms::kSuccess) {
            std::cout << "infer failed." << std::endl;
        }
        inputs.pop_back();
        cv::Mat ccout(config.infer_cout_shape[0], config.infer_cout_shape[1], CV_32FC1, outputs[0].MutableData());
        cv::Mat rout(config.infer_rout_shape[0], config.infer_rout_shape[1], CV_32FC1, outputs[1].MutableData());
        deal_predict(ccout, rout, &config, shape, origin_target_sz, i, resultbox);
        copy_box_two_value(config.pred_box, resultbox[0], resultbox[1]);
        config.flag = judge_failures(resultbox, config.gt_bbox[i]);
        if (config.flag == 0) {
          copy_box_four_value(result_box[i], 2.0, 0.0, 0.0, 0.0);
          template_idx = std::min(i + 5, k - 1);
        } else {
          copy_box(result_box[i], resultbox, 4);
          copy_box(config.box_01, config.pred_box, 4);
        }
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map->insert(std::pair<double, double>(startTime_ms, endTime_ms));
      }
    }
    WriteResult("prediction.txt", result_box, k, FLAGS_dataset_name, dir);
  }
  return 0;
}
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto context = std::make_shared<ms::Context>();
  auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(0);
  context->MutableDeviceInfo().push_back(ascend310_info);
  ms::Graph graph;
  std::cout << "siamRPN file is " << FLAGS_siamRPN_file << std::endl;
  ms::Status ret = ms::Serialization::Load(FLAGS_siamRPN_file, ms::ModelType::kMindIR, &graph);
  if (ret != ms::kSuccess) {
    std::cout << "Load model failed." << std::endl;
    return 1;
  }
  ms::Model siamRPN;
  ret = siamRPN.Build(ms::GraphCell(graph), context);
  if (ret != ms::kSuccess) {
    std::cout << "Build model failed." << std::endl;
    return 1;
  }
  std::vector<ms::MSTensor> model_inputs = siamRPN.GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }
  auto data_set = FLAGS_image_path;
  std::map<double, double> costTime_map;
  std::vector<std::string> dirs;
  dirs = GetAlldir(data_set, FLAGS_dataset_name);
  process_infer(dirs, model_inputs, data_set, &siamRPN, ret, &costTime_map);
  if (ret != ms::kSuccess) {
    std::cout << "process_infer failed." << std::endl;
    return 1;
  }
  std::cout << "process_infer is ok" << std::endl;
  double average = 0.0;
  int inferCount = 0;
  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = 0.0;
    diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average
           << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: " << average
            << "ms of infer_count " << inferCount << std::endl;
  std::string fileName =
      "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
