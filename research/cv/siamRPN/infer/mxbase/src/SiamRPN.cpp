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

#include "SiamRPN.h"

#include <dirent.h>
#include <sys/stat.h>

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <boost/property_tree/json_parser.hpp>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "acl/acl.h"

const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;

const uint32_t MODEL_HEIGHT = 768;
const uint32_t MODEL_WIDTH = 1280;

const uint32_t MODEL_TEMPLATE_HEIGHT = 127;
const uint32_t MODEL_TEMPLATE_WIDTH = 127;

const uint32_t MODEL_DETECTION_HEIGHT = 255;
const uint32_t MODEL_DETECTION_WIDTH = 255;


void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec,
                      const std::string &tensorName) {
  LogInfo << "The shape of " << tensorName << " is as follows:";
  for (size_t i = 0; i < tensorDescVec.size(); ++i) {
    LogInfo << "  Tensor " << i << ":";
    for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
      LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
    }
  }
}

APP_ERROR SiamRPN::Init(const InitParam &initParam) {
  deviceId_ = initParam.deviceId;
  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    return ret;
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    return ret;
  }
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(initParam.modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    return ret;
  }

  PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
  PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");
  post_ = std::make_shared<MxBase::SiamRPNMindsporePost>();
  return APP_ERR_OK;
}

APP_ERROR SiamRPN::DeInit() {
  // dvppWrapper_->DeInit();
  model_->DeInit();
  post_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

void writeMatToFile(const cv::Mat &m, const char *filename) {
  std::ofstream fout(filename);

  if (!fout) {
    std::cout << "File Not Opened" << std::endl;
    return;
  }

  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      fout << m.at<float>(i, j) << "\t";
    }
    fout << std::endl;
  }

  fout.close();
}

APP_ERROR SiamRPN::CVMatToTensorBase(cv::Mat &imageMat,
                                     MxBase::TensorBase &tensorBase,
                                     std::vector<uint32_t> &shape) {
  const uint32_t dataSize =
      imageMat.cols * imageMat.rows * imageMat.channels() * 4;
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
                                   deviceId_);
  MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret =
      MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                  MxBase::TENSOR_DTYPE_FLOAT32);
  return APP_ERR_OK;
}

APP_ERROR SiamRPN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                             std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                      deviceId_);
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }
    outputs.push_back(tensor);
  }
  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  dynamicInfo.batchSize = 1;
  APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);

  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR SiamRPN::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                               MxBase::Tracker &track,
                               MxBase::Config &postConfig, int idx,
                               int total_num, float result_box[][4],
                               int &template_idx) {
  APP_ERROR ret = post_->Process(inputs, postConfig, track, idx, total_num,
                                 result_box, template_idx);
  if (ret != APP_ERR_OK) {
    LogError << "Process failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

std::string RealPath(std::string path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  std::cout << path << " realpath is: " << realPath << std::endl;
  return realPath;
}

DIR *OpenDir(std::string dirName) {
  if (dirName.empty()) {
    std::cout << " dirName is null ! " << std::endl;
    return nullptr;
  }
  std::string realPath = RealPath(dirName);
  struct stat s;
  lstat(realPath.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dirName is not a valid directory !" << std::endl;
    return nullptr;
  }
  DIR *dir = opendir(realPath.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dirName << std::endl;
    return nullptr;
  }
  std::cout << "Successfully opened the dir " << dirName << std::endl;
  return dir;
}

std::vector<std::string> GetAlldir(const std::string &dir_name,
                                   const std::string &data_name) {
  DIR *dir = OpenDir(dir_name + '/' + data_name);
  if (dir == nullptr) {
    return {};
  }
  std::vector<std::string> res;
  if (data_name == "vot2015" || data_name == "vot2016") {
    struct dirent *filename;
    while ((filename = readdir(dir)) != nullptr) {
      std::string d_name = std::string(filename->d_name);
      // get rid of "." and ".."
      if (d_name == "." || d_name == ".." || filename->d_type != DT_DIR)
        continue;
      std::cout << "dirs:" << d_name << std::endl;
      res.emplace_back(d_name);
    }
  }
  return res;
}

std::vector<std::string> GetAllFiles(std::string dirName) {
  struct dirent *filename;
  DIR *dir = OpenDir(dirName);
  if (dir == nullptr) {
    return {};
  }
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string dName = std::string(filename->d_name);
    if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
      continue;
    }
    res.emplace_back(std::string(dirName) + "/" + filename->d_name);
  }
  std::sort(res.begin(), res.end());
  for (auto &f : res) {
  }
  return res;
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

void copy_box(float *box, float *bbox, int num) {
  for (int i = 0; i < num; i++) {
    box[i] = bbox[i];
  }
}
float min_box(float *bbox, int start, int step, int len) {
  float min_value = bbox[start];
  for (int i = start; i < len; i = i + step) {
    if (min_value > bbox[i]) {
      min_value = bbox[i];
    }
  }
  return min_value;
}
float max(float *bbox, int start, int step, int len) {
  float max_value = bbox[start];
  for (int i = start; i < len; i = i + step) {
    if (max_value < bbox[i]) {
      max_value = bbox[i];
    }
  }
  return max_value;
}

void trans_box(float *bbox, float *box) {
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
int WriteResult(const std::string &imageFile, float outputs[][4], int k,
                const std::string &dataset_name, const std::string &seq) {
  std::string homePath;
  homePath = "./result_Files/" + dataset_name + "/" + seq;
  std::string shell = "mkdir ./result_Files";
  std::string shell1 = "mkdir ./result_Files/" + dataset_name;
  std::string shell12 = "mkdir " + homePath;
  system(shell.c_str());
  system(shell1.c_str());
  system(shell12.c_str());
  std::cout << "homePath is " << homePath << std::endl;
  std::string fileName = homePath + '/' + imageFile;
  FILE *fp;
  fp = fopen(fileName.c_str(), "wt");
  for (int i = 0; i < k; i++) {
    fprintf(fp, "%f, ", outputs[i][0]);
    fprintf(fp, "%f, ", outputs[i][1]);
    fprintf(fp, "%f, ", outputs[i][2]);
    fprintf(fp, "%f\n", outputs[i][3]);
  }
  fclose(fp);
  return 0;
}

float max(float a, float b) {
  if (a > b) return a;
  return b;
}
void Pad(const cv::Mat &srcImageMat, cv::Mat &dstImageMat, int left, int bottom, int right, int top) {
    cv::Scalar tempVal = cv::mean(srcImageMat);
    tempVal.val[0] = static_cast<int>(tempVal.val[0]);
    tempVal.val[1] = static_cast<int>(tempVal.val[1]);
    tempVal.val[2] = static_cast<int>(tempVal.val[2]);
    int borderType = cv::BORDER_CONSTANT;
    copyMakeBorder(srcImageMat, dstImageMat, top, bottom, left, right, borderType, tempVal);
}
void Crop(const cv::Mat &img, cv::Mat &crop_img, const std::vector<int> &area) {
    int crop_x1 = std::max(0, area[0]);
    int crop_y1 = std::max(0, area[1]);
    int crop_x2 = std::min(img.cols -1, area[0] + area[2] - 1);
    int crop_y2 = std::min(img.rows - 1, area[1] + area[3] - 1);
    crop_img = img(cv::Range(crop_y1, crop_y2+1), cv::Range(crop_x1, crop_x2 + 1));
}

void ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat, std::vector<int> &size) {
    cv::resize(srcImageMat, dstImageMat, cv::Size(size[0], size[1]));
}

void crop_and_pad(const cv::Mat& img, cv::Mat& dest, float cx, float cy, float original_sz, int im_h, int im_w) {
  float xmin = cx - (original_sz - 1) / 2;
  float xmax = xmin + original_sz - 1;
  float ymin = cy - (original_sz - 1) / 2;
  float ymax = ymin + original_sz - 1;

  int left = static_cast<int>(std::round(max(0, -xmin)));
  int top = static_cast<int>(round(max(0., -ymin)));
  int right = static_cast<int>(round(max(0., xmax - im_w + 1)));
  int bottom = static_cast<int>(round(max(0., ymax - im_h + 1)));

  xmin = static_cast<int>(round(xmin + left));
  xmax = static_cast<int>(round(xmax + left));
  ymin = static_cast<int>(round(ymin + top));
  ymax = static_cast<int>(round(ymax + top));

  std::vector<int> position = { static_cast<int>(xmin), static_cast<int>(ymin), static_cast<int>(xmax-xmin+1),
                               static_cast<int>(ymax- ymin+1) };
  Pad(img, dest, left, bottom, right, top);
  Crop(dest, dest, position);
}
cv::Mat HWC2CHW(cv::Mat& srcImageMat) {
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
  int w = srcImageMat.cols;
  int h = srcImageMat.rows;
  cv::Mat dst = srcImageMat1.reshape(h, w);
  return dst;
}

cv::Mat get_template_Mat_1(std::string file_path, float *box,
                           int resize_template, float context_amount) {
  cv::Mat srcImageMat;
  srcImageMat = cv::imread(file_path, cv::IMREAD_COLOR);
  int h = srcImageMat.cols;
  int w = srcImageMat.rows;
  int cx = box[0] + box[2] / 2 - 0.5;
  int cy = box[1] + box[3] / 2 - 0.5;
  float w_template = box[2] + (box[2] + box[3]) * context_amount;
  float h_template = box[3] + (box[2] + box[3]) * context_amount;
  float s_x = std::sqrt(w_template * h_template);
  std::vector<int> size = {resize_template, resize_template};

  crop_and_pad(srcImageMat, srcImageMat, cx, cy, s_x, w, h);
  ResizeImage(srcImageMat, srcImageMat, size);
  cv::Mat srcImageMat1 = HWC2CHW(srcImageMat);
  return srcImageMat1;
}
cv::Mat get_detection_Mat_1(std::string file_path, float *box,
                            int resize_template, int resize_detection,
                            float context_amount, float *scale_x) {
  cv::Mat srcImageMat;
  srcImageMat = cv::imread(file_path, cv::IMREAD_COLOR);
  int h = srcImageMat.cols;
  int w = srcImageMat.rows;
  int cx = box[0] + box[2] / 2 - 0.5;
  int cy = box[1] + box[3] / 2 - 0.5;
  float w_template = box[2] + (box[2] + box[3]) * context_amount;
  float h_template = box[3] + (box[2] + box[3]) * context_amount;
  float s_x = std::sqrt(w_template * h_template);
  s_x = s_x * resize_detection / resize_template;
  *scale_x = resize_detection / static_cast<float>(s_x);
  std::vector<int> size = {resize_detection, resize_detection};

  crop_and_pad(srcImageMat, srcImageMat, cx, cy, s_x, w, h);
  ResizeImage(srcImageMat, srcImageMat, size);
  cv::Mat srcImageMat1 = HWC2CHW(srcImageMat);

  return srcImageMat1;
}
int abWrite(const cv::Mat &im, const std::string &fname) {
  std::ofstream ouF;
  ouF.open(fname.c_str(), std::ofstream::binary);
  for (int r = 0; r < im.rows; r++) {
    ouF.write(reinterpret_cast<const char *>(im.ptr(r)),
              im.cols * im.elemSize());
  }
  ouF.close();
  return 1;
}
void copy_box_four_value(float* box, float value1, float value2, float value3, float value4) {
  box[0] = value1;
  box[1] = value2;
  box[2] = value3;
  box[3] = value4;
}
APP_ERROR SiamRPN::Process(const std::string &data_set,
                           const std::string &dataset_name) {
  std::vector<std::string> dirs;
  dirs = GetAlldir(data_set, dataset_name);
  MxBase::Tracker track;
  MxBase::Config postConfig;
  postConfig.anchors = MxBase::readMatFromFile("./src/anchors.bin", 1445, 4);
  postConfig.windows = MxBase::readMatFromFile("./src/windows.bin", 1445, 1);
  std::vector<MxBase::TensorBase> inputs = {};
  for (const auto &dir : dirs) {
    std::vector<std::string> images;
    images = GetAllFiles(data_set + '/' + dataset_name + '/' + dir + "/color");
    int k = read_gtBox(
        data_set + '/' + dataset_name + '/' + dir + "/groundtruth.txt",
        track.gt_bbox);
    int template_idx = 0;
    float result_box[k][4];
    std::string image_template;
    std::string image_detection;
    std::vector<uint32_t> template_shape = {3, 127, 127};
    std::vector<uint32_t> detection_shape = {3, 255, 255};
    cv::Mat srcImageMat;
    for (int i = 0; i < static_cast<int>(images.size()); i++) {
      int temp = i;
      if (i == 0) {
        cv::Mat imageMat = cv::imread(images[i], cv::IMREAD_COLOR);
        track.shape[0] = imageMat.rows;
        track.shape[1] = imageMat.cols;
      }
      if (i == template_idx) {
        copy_box(track.bbox, track.gt_bbox[i], 8);
        trans_box(track.bbox, track.box_01);
        track.pos[0] = track.box_01[0] + (track.box_01[2] - 1) / 2;
        track.pos[1] = track.box_01[1] + (track.box_01[3] - 1) / 2;
        track.target_sz[0] = track.box_01[2];
        track.target_sz[1] = track.box_01[3];
        track.origin_target_sz[0] = track.box_01[2];
        track.origin_target_sz[1] = track.box_01[3];
        image_template = images[template_idx];
        srcImageMat =
            get_template_Mat_1(image_template, track.box_01,
                               track.resize_template, track.context_amount);
        MxBase::TensorBase templateTensor;
        APP_ERROR ret =
            CVMatToTensorBase(srcImageMat, templateTensor, template_shape);
        if (ret != APP_ERR_OK) {
          LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
          return ret;
        }
        inputs.clear();
        inputs.push_back(templateTensor);
        copy_box_four_value(result_box[i], 1.0, 0.0, 0.0, 0.0);
      } else if (template_idx > temp) {
        copy_box_four_value(result_box[i], 0.0, 0.0, 0.0, 0.0);
      } else {
        std::vector<MxBase::TensorBase> outputs;
        image_detection = images[i];
        std::cout << "Start predict input files:" << image_template
                  << image_detection << std::endl;
        cv::Mat decImageMat = get_detection_Mat_1(
            image_detection, track.box_01, track.resize_template,
            track.resize_detection, track.context_amount, &track.scale_x);
        MxBase::TensorBase detectionTensor;
        APP_ERROR ret1 =
            CVMatToTensorBase(decImageMat, detectionTensor, detection_shape);
        if (ret1 != APP_ERR_OK) {
          LogError << "CVMatToTensorBase failed, ret=" << ret1 << ".";
          return ret1;
        }
        inputs.push_back(detectionTensor);
        APP_ERROR ret2 = Inference(inputs, outputs);
        if (ret2 != APP_ERR_OK) {
          LogError << "postProcess failed, ret=" << ret2 << ".";
          return ret2;
        }
        inputs.pop_back();
        APP_ERROR ret3 = PostProcess(outputs, track, postConfig, i, k,
                                    result_box, template_idx);
        if (ret3 != APP_ERR_OK) {
          LogError << "postProcess failed, ret=" << ret3 << ".";
          return ret3;
        }
      }
      WriteResult("prediction.txt", result_box, k, dataset_name, dir);
    }
  }
  return APP_ERR_OK;
}
