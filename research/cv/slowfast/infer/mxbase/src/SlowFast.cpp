/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
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

#include "SlowFast.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using MxBase::DeviceManager;
using MxBase::DvppWrapper;
using MxBase::DynamicInfo;
using MxBase::DynamicType;
using MxBase::MemoryData;
using MxBase::MemoryHelper;
using MxBase::ModelInferenceProcessor;
using MxBase::TensorBase;
using MxBase::TensorContext;

SLOWFAST::SLOWFAST(const uint32_t &deviceId, const std::string &modelPath,
                   const std::string &datadir)
    : deviceId_(deviceId) {
  LogInfo << "SLOWFAST infer Construct!!!";

  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    exit(-1);
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(deviceId);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    exit(-1);
  }

  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    exit(-1);
  }
  DATADIR = datadir;
  FRAME_LIST_DIR = DATADIR + ANN_DIR;
  ANNOTATION_DIR = DATADIR + ANN_DIR;
  FRAME_DIR = DATADIR + FRA_DIR;
}

SLOWFAST::~SLOWFAST() {
  model_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
}

void SLOWFAST::ReadImage(std::string imgPath, cv::Mat &imageMat) {
  imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);

  cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
}

void SLOWFAST::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
  cv::resize(srcImageMat, dstImageMat, cv::Size(_scale_width, _scale_height), 0,
             0, cv::INTER_LINEAR);
}
void SLOWFAST::LoadData1() {
  // get _image_paths and _video_idx_to_name
  _image_paths.resize(0);
  _video_idx_to_name.resize(0);
  std::vector<std::string> _image_paths_tmp;
  _image_paths_tmp.resize(0);
  _video_idx_to_name.push_back("tmp");
  std::string filename = FRAME_LIST_DIR + "/" + TEST_LISTS;
  std::ifstream fin;
  fin.open(filename);
  std::string t1, t2, t3, t4, t5;
  fin >> t1 >> t2 >> t3 >> t4 >> t5;
  while (!fin.eof()) {
    fin >> t1 >> t2 >> t3 >> t4 >> t5;
    if (t1 != _video_idx_to_name.back()) {
      _image_paths.push_back(_image_paths_tmp);
      _image_paths_tmp.resize(0);
      _video_idx_to_name.push_back(t1);
    }
    _image_paths_tmp.push_back(t4);
  }
  fin.close();
  _image_paths.push_back(_image_paths_tmp);
  _image_paths.erase(_image_paths.begin());
  _video_idx_to_name.erase(_video_idx_to_name.begin());
  return;
}
void SLOWFAST::LoadData2() {
  // Loading annotations for boxes and labels.
  std::ifstream fin;
  //   std::vector<std::map<int, std::vector<std::vector<float>>>> boxs_all;
  boxs_all.resize(0);
  std::map<int, std::vector<std::vector<float>>> box_for_video;
  std::vector<std::vector<float>> boxs;
  std::vector<float> box_each;
  std::string video_name_tmp = "1j20qq1JyX4";
  int frame_tmp = 901;
  std::string ann_filenames = ANNOTATION_DIR + "/" + TEST_PREDICT_BOX_LISTS;
  float detect_thresh = DETECTION_SCORE_THRESH;
  int boxes_sample_rate = 1;
  int box_end = 6;
  fin.open(ann_filenames);
  std::string linestr;
  while (getline(fin, linestr)) {
    if (linestr.empty()) {
      continue;
    }
    std::vector<std::string> strvec;
    std::string s;
    std::stringstream ss(linestr);

    while (getline(ss, s, ',')) {
      strvec.push_back(s);
    }
    float score = static_cast<float>(stod(strvec[7]));
    if (score < detect_thresh) {
      continue;
    }
    std::string video_name = strvec[0];
    int frame_sec = static_cast<int>(stod(strvec[1]));
    if (frame_sec % boxes_sample_rate != 0) {
      continue;
    }
    box_each.resize(0);
    for (int ii = 2; ii < box_end; ii++) {
      box_each.push_back(static_cast<float>(stod(strvec[ii])));
    }
    box_each.push_back(-1);

    if (frame_sec != frame_tmp) {
      box_for_video[frame_tmp] = boxs;
      boxs.resize(0);
      frame_tmp = frame_sec;
    }
    boxs.push_back(box_each);

    if (video_name != video_name_tmp) {
      boxs_all.push_back(box_for_video);
      box_for_video.erase(box_for_video.begin(), box_for_video.end());
      video_name_tmp = video_name;
    }
  }
  fin.close();
  box_for_video[frame_tmp] = boxs;
  boxs_all.push_back(box_for_video);
  return;
}
void SLOWFAST::LoadData3() {
  // Get indices of keyframes and corresponding boxes and labels.
  _keyframe_indices.resize(0);
  _keyframe_boxes_and_labels.resize(0);
  int size_tmp = boxs_all.size();
  for (int ii = 0; ii < size_tmp; ii++) {
    int sec_idx = 0;
    std::vector<std::vector<std::vector<float>>> box_temp;
    for (auto it : boxs_all[ii]) {
      int sec = it.first;
      if (frame_min <= sec && sec <= frame_max) {
        if (!it.second.empty()) {
          _keyframe_indices.push_back({ii, sec_idx, sec, (sec - 900) * 30});
          box_temp.push_back(it.second);
          sec_idx++;
        }
      }
    }

    _keyframe_boxes_and_labels.push_back(box_temp);
  }
  return;
}
void SLOWFAST::LoadData() {
  LoadData1();
  LoadData2();
  LoadData3();
}

void SLOWFAST::GetData(int idx) {
  int video_idx = _keyframe_indices[idx][0];
  int sec_idx = _keyframe_indices[idx][1];
  int sec = _keyframe_indices[idx][2];
  int center_idx = _keyframe_indices[idx][3];
  //# Get the frame idxs for current clip.
  std::vector<int> seq = get_sequence(center_idx, _seq_len / 2, _sample_rate,
                                      _image_paths[video_idx].size());
  std::vector<std::vector<float>> clip_label_list =
      _keyframe_boxes_and_labels[video_idx][sec_idx];
  // # Get boxes and labels for current clip.
  std::vector<std::vector<float>> boxes;
  std::vector<int> labels;
  for (auto ii : clip_label_list) {
    std::vector<float> tmp;
    tmp.assign(ii.begin(), ii.begin() + 4);
    boxes.push_back(tmp);
    labels.push_back(static_cast<int>(ii[4]));
  }
  std::vector<std::vector<float>> ori_boxes(boxes);
  //# Load images of current clip.
  std::vector<std::string> image_paths;
  for (size_t ii = 0; ii < seq.size(); ii++) {
    image_paths.push_back(_image_paths[video_idx][seq[ii]]);
  }
  std::vector<cv::Mat> imgs;
  for (auto path : image_paths) {
    cv::Mat imageMat;
    path = FRAME_DIR + "/" + path;
    ReadImage(path, imageMat);
    imgs.push_back(imageMat);
  }

  //# Preprocess images and boxes
  int height = imgs[0].rows;
  int width = imgs[0].cols;
  boxes = ResizeBox(boxes, height, width);

  for (size_t ii = 0; ii < imgs.size(); ii++) {
    ResizeImage(imgs[ii], imgs[ii]);
  }

  for (size_t ii = 0; ii < imgs.size(); ii++) {
    imgs[ii].convertTo(imgs[ii], CV_32FC3);

    imgs[ii].convertTo(imgs[ii], -1, 1.0 / 255, 0);
    float a = -0.45;
    imgs[ii].convertTo(imgs[ii], -1, 1.0, a);
    float b = 1 / 0.225;
    imgs[ii].convertTo(imgs[ii], -1, b, 0);
  }

  //    # Construct label arrays.

  slow_pathway.resize(0);
  fast_pathway.resize(0);
  for (size_t ii = 0; ii < imgs.size(); ii++) {
    fast_pathway.push_back(imgs[ii]);
    if (ii % ALPHA == 0) {
      slow_pathway.push_back(imgs[ii]);
    }
  }
  std::vector<std::vector<int>> metadata;
  for (size_t ii = 0; ii < boxes.size(); ii++) {
    metadata.push_back({video_idx, sec});
  }
  padded_boxes = pad2max_float(boxes);
  mask.resize(0);
  for (int ii = 0; ii < MAX_NUM_BOXES_PER_FRAME; ii++) {
    mask.push_back(0);
  }
  for (size_t ii = 0; ii < boxes.size(); ii++) {
    mask[ii] = 1;
  }
  padded_ori_boxes = pad2max_float(ori_boxes);
  padded_metadata = pad2max_int(metadata);
}
std::vector<std::vector<float>> SLOWFAST::ResizeBox(
    std::vector<std::vector<float>> boxes, int height, int width) {
  for (size_t ii = 0; ii < boxes.size(); ii++) {
    boxes[ii][0] *= width;
    if (boxes[ii][0] > (width - 1)) {
      boxes[ii][0] = width - 1;
    }
    boxes[ii][2] *= width;
    if (boxes[ii][2] > (width - 1)) {
      boxes[ii][2] = width - 1;
    }
    boxes[ii][1] *= height;
    if (boxes[ii][1] > (height - 1)) {
      boxes[ii][1] = height - 1;
    }
    boxes[ii][3] *= height;
    if (boxes[ii][3] > (height - 1)) {
      boxes[ii][3] = height - 1;
    }
  }

  for (size_t ii = 0; ii < boxes.size(); ii++) {
    boxes[ii][0] = boxes[ii][0] * _scale_height / height;
    boxes[ii][2] = boxes[ii][2] * _scale_height / height;
    boxes[ii][1] = boxes[ii][1] * _scale_width / width;
    boxes[ii][3] = boxes[ii][3] * _scale_width / width;
  }

  for (size_t ii = 0; ii < boxes.size(); ii++) {
    if (boxes[ii][0] > (_scale_width - 1)) {
      boxes[ii][0] = _scale_width - 1;
    }

    if (boxes[ii][2] > (_scale_width - 1)) {
      boxes[ii][2] = _scale_width - 1;
    }

    if (boxes[ii][1] > (_scale_height - 1)) {
      boxes[ii][1] = _scale_height - 1;
    }

    if (boxes[ii][3] > (_scale_height - 1)) {
      boxes[ii][3] = _scale_height - 1;
    }
  }

  return boxes;
}
std::vector<int> SLOWFAST::get_sequence(int center_idx, int half_len,
                                        int sample_rate, int num_frames) {
  // Sample frames among the corresponding clip.

  // Args:
  //     center_idx (int): center frame idx for current clip
  //     half_len (int): half of the clip length
  //     sample_rate (int): sampling rate for sampling frames inside of the clip
  //     num_frames (int): number of expected sampled frames

  // Returns:
  //     seq (list): list of indexes of sampled frames in this clip.

  std::vector<int> seq;
  for (int ii = center_idx - half_len; ii < center_idx + half_len; ii += 2) {
    seq.push_back(ii);
  }
  for (size_t ii = 0; ii < seq.size(); ii++) {
    if (seq[ii] < 0) {
      seq[ii] = 0;
    } else if (seq[ii] >= num_frames) {
      seq[ii] = num_frames - 1;
    }
  }
  return seq;
}

std::vector<std::vector<float>> SLOWFAST::pad2max_float(
    std::vector<std::vector<float>> data) {
  std::vector<float> tmp(data[0].size(), 0);
  std::vector<std::vector<float>> padded_data(MAX_NUM_BOXES_PER_FRAME, tmp);
  int size_tmp = data.size();
  for (int ii = 0; ii < size_tmp; ii++) {
    padded_data[ii] = data[ii];
  }
  return padded_data;
}
std::vector<std::vector<int>> SLOWFAST::pad2max_int(
    std::vector<std::vector<int>> data) {
  std::vector<int> tmp(data[0].size(), 0);
  std::vector<std::vector<int>> padded_data(MAX_NUM_BOXES_PER_FRAME, tmp);
  int size_tmp = data.size();
  for (int ii = 0; ii < size_tmp; ii++) {
    padded_data[ii] = data[ii];
  }
  return padded_data;
}
int SLOWFAST::get_max() { return _keyframe_indices.size(); }
int SLOWFAST::get_batch_size() { return BATCH_SIZE; }
std::vector<cv::Mat> SLOWFAST::get_slow_pathway() { return slow_pathway; }
std::vector<cv::Mat> SLOWFAST::get_fast_pathway() { return fast_pathway; }
std::vector<std::vector<float>> SLOWFAST::get_padded_boxes() {
  return padded_boxes;
}

APP_ERROR SLOWFAST::VectorToTensorBase_mat(
    const std::vector<std::vector<cv::Mat>> &input, int idx,
    MxBase::TensorBase *tensorBase) {
  uint32_t dataSize = 1;

  std::vector<uint32_t> shape = {};
  for (size_t j = 0; j < modelDesc_.inputTensors[idx].tensorDims.size(); j++) {
    shape.push_back((uint32_t)modelDesc_.inputTensors[idx].tensorDims[j]);
  }
  for (uint32_t s = 0; s < shape.size(); ++s) {
    dataSize *= shape[s];
  }

  float *metaFeatureData = new float[dataSize];
  int bs_max = input.size();
  int chan = input[0][0].channels();
  int n_max = input[0].size();
  int row_max = _scale_height;
  int col_max = _scale_width;

  uint32_t idx_ = 0;

  for (int bs = 0; bs < bs_max; bs++) {
    for (int c = 0; c < chan; c++) {
      for (int n = 0; n < n_max; n++) {
        for (int row = 0; row < row_max; row++) {
          for (int col = 0; col < col_max; col++) {
            metaFeatureData[idx_++] = input[bs][n].ptr<float>(row)[col * 3 + c];
          }
        }
      }
    }
  }
  MxBase::MemoryData memoryDataDst(
      dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
  MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData),
                                   dataSize * 4,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret =
      MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }

  *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                   MxBase::TENSOR_DTYPE_FLOAT32);
  delete metaFeatureData;
  return APP_ERR_OK;
}
APP_ERROR SLOWFAST::VectorToTensorBase_float(
    const std::vector<std::vector<std::vector<float>>> &input, int idx,
    MxBase::TensorBase *tensorBase) {
  uint32_t dataSize = 1;

  std::vector<uint32_t> shape = {};
  for (size_t j = 0; j < modelDesc_.inputTensors[idx].tensorDims.size(); j++) {
    shape.push_back((uint32_t)modelDesc_.inputTensors[idx].tensorDims[j]);
  }
  for (uint32_t s = 0; s < shape.size(); ++s) {
    dataSize *= shape[s];
  }

  float *metaFeatureData = new float[dataSize];
  int bs_max = input.size();
  int n_max = input[0].size();
  int d_max = input[0][0].size();

  uint32_t idx_ = 0;

  for (int bs = 0; bs < bs_max; bs++) {
    for (int n = 0; n < n_max; n++) {
      for (int d = 0; d < d_max; d++) {
        metaFeatureData[idx_++] = input[bs][n][d];
      }
    }
  }

  MxBase::MemoryData memoryDataDst(
      dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
  MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData),
                                   dataSize * 4,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret =
      MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }

  *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                   MxBase::TENSOR_DTYPE_FLOAT32);
  delete metaFeatureData;
  return APP_ERR_OK;
}

APP_ERROR SLOWFAST::Inference(const std::vector<MxBase::TensorBase> &inputs,
                              std::vector<MxBase::TensorBase> *outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); i++) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); j++) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i],
                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                              deviceId_);
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }
    (*outputs).push_back(tensor);
  }
  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  auto startTime = std::chrono::high_resolution_clock::now();
  APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  inferCostTimeMilliSec += costMs;
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR SLOWFAST::SaveInferResult(
    std::vector<float> *batchFeaturePaths,
    const std::vector<MxBase::TensorBase> &inputs) {
  for (auto retTensor : inputs) {
    std::vector<uint32_t> shape = retTensor.GetShape();
    uint32_t N = shape[0];
    uint32_t C = shape[1];
    if (!retTensor.IsHost()) {
      retTensor.ToHost();
    }
    void *data = retTensor.GetBuffer();

    for (uint32_t i = 0; i < N; i++) {
      for (uint32_t j = 0; j < C; j++) {
        float value = *(reinterpret_cast<float *>(data) + i * C + j);
        batchFeaturePaths->emplace_back(value);
      }
    }
  }
  return APP_ERR_OK;
}

APP_ERROR SLOWFAST::Process(
    const std::vector<std::vector<cv::Mat>> &input1,
    const std::vector<std::vector<cv::Mat>> &input2,
    const std::vector<std::vector<std::vector<float>>> &input3,
    const std::vector<float> *output) {
  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs;
  std::vector<float> batchFeaturePaths;
  MxBase::TensorBase tensorBase;
  auto ret0 = VectorToTensorBase_mat(input1, 0, &tensorBase);
  if (ret0 != APP_ERR_OK) {
    LogError << "ToTensorBase failed, ret=" << ret0 << ".";
    return ret0;
  }
  inputs.push_back(tensorBase);
  auto ret1 = VectorToTensorBase_mat(input2, 1, &tensorBase);
  if (ret1 != APP_ERR_OK) {
    LogError << "ToTensorBase failed, ret=" << ret1 << ".";
    return ret1;
  }
  inputs.push_back(tensorBase);
  auto ret2 = VectorToTensorBase_float(input3, 2, &tensorBase);
  if (ret2 != APP_ERR_OK) {
    LogError << "ToTensorBase failed, ret=" << ret2 << ".";
    return ret2;
  }
  inputs.push_back(tensorBase);
  auto startTime = std::chrono::high_resolution_clock::now();
  auto ret3 = Inference(inputs, &outputs);

  auto endTime = std::chrono::high_resolution_clock::now();
  double costMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  inferCostTimeMilliSec += costMs;
  if (ret3 != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret3 << ".";
    return ret3;
  }

  auto ret4 = SaveInferResult(&batchFeaturePaths, outputs);
  if (ret4 != APP_ERR_OK) {
    LogError << "Save model infer results into file failed. ret = " << ret4
             << ".";
    return ret4;
  }

  std::vector<std::vector<float>> preds;
  preds.resize(0);
  std::vector<std::vector<int>> metadata;
  metadata.resize(0);
  std::vector<float> preds_tmp;
  preds_tmp.resize(0);
  size_t ii_max = mask.size();
  size_t jj_max = _num_classes;
  size_t idx = 0;
  for (size_t ii = 0; ii < ii_max; ii++) {
    for (size_t jj = 0; jj < jj_max; jj++) {
      preds_tmp.push_back(batchFeaturePaths[idx++]);
    }
    if (mask[ii] == 1) {
      preds.push_back(preds_tmp);
      metadata.push_back(padded_metadata[ii]);
    }
    preds_tmp.resize(0);
  }
  for (size_t ii = 0; ii < padded_ori_boxes.size(); ii++) {
    padded_ori_boxes[ii].insert(padded_ori_boxes[ii].begin(), 0);
  }

  for (size_t ii = 0; ii < preds.size(); ii++) {
    preds_final.push_back(preds[ii]);
    ori_boxes_final.push_back(padded_ori_boxes[ii]);
    metadata_final.push_back(metadata[ii]);
  }

  return APP_ERR_OK;
}
std::vector<std::string> SLOWFAST::ReadExcluded_keys() {
  // read_exclusions
  std::vector<std::string> excluded_keys;
  std::string filename = ANNOTATION_DIR + "/" + EXCLUSION_FILE;
  std::ifstream fin;
  fin.open(filename);
  std::string linestr;
  while (getline(fin, linestr)) {
    if (linestr.empty()) {
      continue;
    }
    excluded_keys.push_back(linestr);
  }
  fin.close();
  return excluded_keys;
}
std::vector<int> SLOWFAST::ReadClass() {
  // read_labelmap
  std::string filename = ANNOTATION_DIR + "/" + LABEL_MAP_FILE;
  std::vector<int> class_whitelist;
  std::string name;
  std::ifstream fin;
  std::string linestr;
  fin.open(filename);
  while (getline(fin, linestr)) {
    if (linestr.empty()) {
      continue;
    }
    int num_tmp_1 = linestr.find("  name:");
    int num_tmp_2 = linestr.find("  id:");
    if (num_tmp_1 > -1) {
      int i1 = linestr.find("\"");
      int i2 = linestr.find("\"", i1 + 1);
      name = linestr.substr(i1 + 1, i2 - 2);
    } else if (num_tmp_2 > -1) {
      int i1 = linestr.find("id:");
      std::string res = linestr.substr(i1 + 1);
      i1 = res.find(" ");
      res = res.substr(i1 + 1);
      int class_id = atoi(res.c_str());
      class_whitelist.push_back(class_id);
    }
  }
  fin.close();
  return class_whitelist;
}
APP_ERROR SLOWFAST::post_process() {
  std::vector<std::string> excluded_keys = ReadExcluded_keys();
  std::vector<int> class_whitelist = ReadClass();

  // groundtruth
  std::map<std::string, std::vector<std::vector<float>>> boxes_groundtruth;
  std::map<std::string, std::vector<float>> labels_groundtruth;
  std::map<std::string, std::vector<float>> scores_groundtruth;
  std::string filename = ANNOTATION_DIR + "/" + GROUNDTRUTH_FILE;
  std::ifstream fin;
  std::string linestr;
  fin.open(filename);
  while (getline(fin, linestr)) {
    if (linestr.empty()) {
      continue;
    }
    std::vector<std::string> strvec;
    std::string s;
    std::stringstream ss(linestr);

    while (getline(ss, s, ',')) {
      strvec.push_back(s);
    }
    std::string image_key = strvec[0] + "," + strvec[1];
    float x1 = static_cast<float>(stod(strvec[2]));
    float y1 = static_cast<float>(stod(strvec[3]));
    float x2 = static_cast<float>(stod(strvec[4]));
    float y2 = static_cast<float>(stod(strvec[5]));
    std::vector<float> tmp = {y1, x1, y2, x2};
    int action_id = static_cast<int>(stod(strvec[6]));
    std::vector<int>::iterator it =
        std::find(class_whitelist.begin(), class_whitelist.end(), action_id);
    if (it == class_whitelist.end()) {
      continue;
    }
    boxes_groundtruth[image_key].push_back(tmp);
    labels_groundtruth[image_key].push_back(action_id);
    scores_groundtruth[image_key].push_back(1.0);
  }
  fin.close();
  // detections
  std::map<std::string, std::vector<std::vector<float>>> *boxes_detections =
      new std::map<std::string, std::vector<std::vector<float>>>;
  std::map<std::string, std::vector<float>> *labels_detections =
      new std::map<std::string, std::vector<float>>;
  std::map<std::string, std::vector<float>> *scores_detections =
      new std::map<std::string, std::vector<float>>;
  for (size_t ii = 0; ii < preds_final.size(); ii++) {
    int video_idx = metadata_final.at(ii)[0];
    int sec = metadata_final.at(ii)[1];
    std::string video = _video_idx_to_name[video_idx];
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%04d", sec);
    std::string key = video + "," + buffer;
    std::vector<float> batch_box;
    batch_box.push_back(ori_boxes_final.at(ii)[2]);
    batch_box.push_back(ori_boxes_final.at(ii)[1]);
    batch_box.push_back(ori_boxes_final.at(ii)[4]);
    batch_box.push_back(ori_boxes_final.at(ii)[3]);
    for (size_t jj = 0; jj < preds_final.at(ii).size(); jj++) {
      std::vector<int>::iterator it =
          std::find(class_whitelist.begin(), class_whitelist.end(), jj + 1);
      if (it != class_whitelist.end()) {
        auto it_find_1 = scores_detections->find(key);
        if (it_find_1 == scores_detections->end()) {
          std::vector<float> tmp;
          scores_detections->insert(std::make_pair(key, tmp));
        }
        scores_detections->at(key).push_back(preds_final.at(ii)[jj]);

        auto it_find_2 = labels_detections->find(key);
        if (it_find_2 == labels_detections->end()) {
          std::vector<float> tmp;
          labels_detections->insert(std::make_pair(key, tmp));
        }
        labels_detections->at(key).push_back(jj + 1);

        auto it_find_3 = boxes_detections->find(key);
        if (it_find_3 == boxes_detections->end()) {
          std::vector<std::vector<float>> tmp;
          boxes_detections->insert(std::make_pair(key, tmp));
        }
        boxes_detections->at(key).push_back(batch_box);
      }
    }
  }

  LogInfo << "Evaluating with " << boxes_groundtruth.size()
          << " unique GT frames.";
  LogInfo << "Evaluating with " << boxes_detections->size()
          << " unique detection frames.";

  filename = OUTPUT_DIR + "detections_latest_mxbase.csv";
  write_results(*boxes_detections, *labels_detections, *scores_detections,
                filename);
  filename = OUTPUT_DIR + "groundtruth_latest_mxbase.csv";
  write_results(boxes_groundtruth, labels_groundtruth, scores_groundtruth,
                filename);
  return APP_ERR_OK;
}
void SLOWFAST::write_results(
    std::map<std::string, std::vector<std::vector<float>>> boxes,
    std::map<std::string, std::vector<float>> labels,
    std::map<std::string, std::vector<float>> scores, std::string filename) {
  std::ofstream fout(filename);
  for (auto it = boxes.begin(); it != boxes.end(); it++) {
    std::string key = it->first;
    for (size_t ii = 0; ii < boxes[key].size(); ii++) {
      std::vector<float> box = boxes[key][ii];
      int label = labels[key][ii];
      float score = scores[key][ii];
      char buffer[200];
      snprintf(buffer, sizeof(buffer), "%s,%.03f,%.03f,%.03f,%.03f,%d,%.04f\n",
               key.c_str(), box[1], box[0], box[3], box[2], label, score);
      fout << buffer;
    }
  }
  fout.close();
  return;
}
