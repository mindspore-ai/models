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
#include "SiamFCBase.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"
std::vector<std::string> all_videos = {
    "Basketball",   "Bolt",      "Boy",       "Car4",     "CarDark",
    "CarScale",     "Coke",      "Couple",    "Crossing", "David",
    "David2",       "David3",    "Deer",      "Dog1",     "Doll",
    "Dudek",        "FaceOcc1",  "FaceOcc2",  "Fish",     "FleetFace",
    "Football",     "Football1", "Football1", "Freeman1", "Freeman3",
    "Freeman4",     "Girl",      "Ironman",   "Jogging",  "Jumping",
    "Lemming",      "Liquor",    "Matrix",    "Mhyang",   "MotorRolling",
    "MountainBike", "Shaking",   "Singer1",   "Singer2",  "Skating1",
    "Skiing",       "Soccer",    "Subway",    "Suv",      "Sylvester",
    "Tiger1",       "Tiger2",    "Trellis",   "Walking",  "Walking2",
    "Woman"};
APP_ERROR SiamFCBase::Init(const initParam& init) {
  deviceId_ = init.deviceId;
  APP_ERROR ret1 = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret1 != APP_ERR_OK) {
    std::cout << "Init devices failed, ret=" << ret1 << "." << std::endl;
    return ret1;
  }
  APP_ERROR ret2 =
      MxBase::TensorContext::GetInstance()->SetContext(init.deviceId);
  if (ret2 != APP_ERR_OK) {
    std::cout << "Set context failed, ret=" << ret2 << "." << std::endl;
    return ret2;
  }
  model_1 = std::make_shared<MxBase::ModelInferenceProcessor>();
  APP_ERROR ret6 = model_1->Init(init.modelPath1, modelDesc_1);
  if (ret6 != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor1 init failed, ret=" << ret6 << ".";
    return ret6;
  }
  model_2 = std::make_shared<MxBase::ModelInferenceProcessor>();
  APP_ERROR ret7 = model_2->Init(init.modelPath2, modelDesc_2);
  if (ret7 != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor2 init failed, ret=" << ret7 << ".";
    return ret7;
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::DeInit() {
  model_1->DeInit();
  model_2->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}
std::string SiamFCBase::RealPath(const std::string& path) {
  char realPathMem[DATA_LENTH] = {0};
  char* realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  std::cout << path << " realpath is: " << realPath << std::endl;
  return realPath;
}
APP_ERROR SiamFCBase::OpenDir(const std::string& dirName, DIR*& dir) {
  if (dirName.empty()) {
    std::cout << " dirName is null ! " << std::endl;
    return 0;
  }
  std::string realPath = RealPath(dirName);
  struct stat s;
  lstat(realPath.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dirName is not a valid directory !" << std::endl;
    return 0;
  }
  dir = opendir(realPath.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dirName << std::endl;
    return 0;
  }
  std::cout << "Successfully opened the dir " << dirName << std::endl;
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::GetAllFiles(const std::string& dirName, const std::string& seq_name,
                                  param& initParam) {
  struct dirent* filename;
  std::string seqName = std::string(dirName) + "/" + seq_name + "/img";
  DIR* dir;
  APP_ERROR ret = OpenDir(seqName, dir);
  if (ret != APP_ERR_OK) {
    std::cout << GetError(ret) << "OpenDir failed.";
    return ret;
  }
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string dName = std::string(filename->d_name);
    if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
      continue;
    }
    res.emplace_back(std::string(dirName) + "/" + seq_name + "/img/" +
                     filename->d_name);
  }
  std::sort(res.begin(), res.end());
  std::cout << "res_all: " << res.size() << std::endl;
  if (seq_name == "David") {
    initParam.all_files.resize(471);
    std::copy(res.begin() + 299, res.begin() + 770,
              initParam.all_files.begin());
  } else if (seq_name == "Football1") {
    initParam.all_files.resize(74);
    std::copy(res.begin(), res.begin() + 74, initParam.all_files.begin());
  } else if (seq_name == "Freeman3") {
    initParam.all_files.resize(460);
    std::copy(res.begin(), res.begin() + 460, initParam.all_files.begin());
  } else if (seq_name == "Freeman4") {
    initParam.all_files.resize(283);
    std::copy(res.begin(), res.begin() + 283, initParam.all_files.begin());
  } else if (seq_name == "Diving") {
    initParam.all_files.resize(215);
    std::copy(res.begin(), res.begin() + 215, initParam.all_files.begin());
  } else {
    for (size_t i = 0; i < res.size(); i++) {
      initParam.all_files.emplace_back(res[i]);
    }
    std::cout << "image num = " << initParam.all_files.size() << std::endl;
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::Getpos(const std::string& dirName, param& initParam) {
  std::ifstream infile;
  infile.open(dirName.c_str());
  std::string s;
  getline(infile, s);
  std::stringstream ss;
  ss << s;
  double temp;
  while (ss >> temp) {
    initParam.box.push_back(temp);
    if (ss.peek() == ',' || ss.peek() == ' ' || ss.peek() == '\t') {
      ss.ignore();
    }
  }
  infile.close();
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::CropAndPad(const cv::Mat& img, cv::Mat& target, float cx,
                                 float cy, float size_z, float s_z) {
  float xmin = cx - s_z / 2;
  float xmax = cx + s_z / 2;
  float ymin = cy - s_z / 2;
  float ymax = cy + s_z / 2;
  int w = img.cols;
  int h = img.rows;
  int left = 0;
  int right = 0;
  int top = 0;
  int bottom = 0;

  if (xmin < 0) left = static_cast<int>(abs(xmin));
  if (xmax > w) right = static_cast<int>(xmax - w);
  if (ymin < 0) top = static_cast<int>(abs(ymin));
  if (ymax > h) bottom = static_cast<int>(ymax - h);

  xmin = std::max(0, static_cast<int>(xmin));
  xmax = std::min(w, static_cast<int>(xmax));
  ymin = std::max(0, static_cast<int>(ymin));
  ymax = std::min(h, static_cast<int>(ymax));

  target = img(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
  if (left != 0 || right != 0 || top != 0 || bottom != 0) {
    cv::Scalar tempVal = cv::mean(img);
    tempVal.val[0] = static_cast<int>(tempVal.val[0]);
    tempVal.val[1] = static_cast<int>(tempVal.val[1]);
    tempVal.val[2] = static_cast<int>(tempVal.val[2]);
    cv::copyMakeBorder(target, target, top, bottom, left, right,
                       cv::BORDER_CONSTANT, tempVal);
  }
  if (size_z != s_z) {
    cv::resize(target, target, cv::Size(size_z, size_z));
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::HWC2CHW(const cv::Mat& src, cv::Mat& dst,
                              size_t resize_detection) {
  std::vector<float> dst_data;
  std::vector<cv::Mat> bgrChannels(3);
  cv::split(src, bgrChannels);
  for (size_t i = 0; i < bgrChannels.size(); i++) {
    std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
    dst_data.insert(dst_data.end(), data.begin(), data.end());
  }
  cv::Mat srcMat;
  srcMat = cv::Mat(dst_data, true);
  dst = srcMat.reshape(3, resize_detection);
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::PreTreatMent(cv::Mat src, cv::Mat& target, param config,
                                   int size, double s_x) {
  cv::Mat cropImg;
  APP_ERROR ret = CropAndPad(src, cropImg, config.target_position[0],
                             config.target_position[1], size, s_x);
  if (ret != APP_ERR_OK) {
    std::cout << GetError(ret) << "CropAndPad failed.";
    return ret;
  }
  cv::Mat exemplar_FLOAT;
  cv::cvtColor(cropImg, cropImg, cv::COLOR_BGR2RGB);
  cropImg.convertTo(exemplar_FLOAT, CV_32FC3);
  APP_ERROR ret_change = HWC2CHW(exemplar_FLOAT, target, size);
  if (ret_change != APP_ERR_OK) {
    std::cout << GetError(ret) << "Memory malloc failed.";
    return ret_change;
  }
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::InitPosition(param& config, const std::string& temp_video) {
  APP_ERROR ret_getfiles = GetAllFiles(config.dataset_path, temp_video, config);
  if (ret_getfiles != APP_ERR_OK) {
    std::cout << "GetAllFiles failed, ret =" << ret_getfiles << "." << std::endl;
    return ret_getfiles;
  }
  APP_ERROR ret_initpos = Getpos(config.dataset_path_txt, config);
  if (ret_initpos != APP_ERR_OK) {
    std::cout << "Getpos failed, ret =" << ret_initpos << "." << std::endl;
    return ret_getfiles;
  }
  config.size_s = config.all_files.size();
  config.init_x = config.box[0] - 1;
  config.init_y = config.box[1] - 1;
  config.init_w = config.box[2];
  config.init_h = config.box[3];
  config.target_position[0] = config.init_x + (config.init_w - 1) / 2;
  config.target_position[1] = config.init_y + (config.init_h - 1) / 2;
  config.target_sz[0] = config.init_w;
  config.target_sz[1] = config.init_h;
  config.wc_z = config.init_w + 0.5 * (config.init_w + config.init_h);
  config.hc_z = config.init_h + 0.5 * (config.init_w + config.init_h);
  config.s_z = sqrt(config.wc_z * config.hc_z);
  config.scale_z = 127 / config.s_z;
  config.s_x = config.s_z + (255 - 127) / config.scale_z;
  config.min_s_x = 0.2 * config.s_x;
  config.max_s_x = 5 * config.s_x;
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::GetPath(param& config, const std::string& temp_video,
                              int jogging_count, const std::string& seq_root_path,
                              const std::string& code_path) {
  config.dataset_path = seq_root_path;
  config.dataset_path_txt =
      seq_root_path + "/" + temp_video + "/" + "groundtruth_rect.txt";
  config.record_name =
      code_path + "/results/OTB2013/SiamFC/" + temp_video + ".txt";
  config.record_times =
      code_path + "/results/OTB2013/SiamFC/times/" + temp_video + "_time.txt";
  if (temp_video == "Jogging") {
    auto jogging_path = seq_root_path + "/" + temp_video + "/" +
                        "groundtruth_rect" + "." +
                        std::to_string(jogging_count) + ".txt";
    auto jogging_record = code_path + "/results/OTB2013/SiamFC/" + temp_video +
                          "." + std::to_string(jogging_count) + ".txt";
    config.dataset_path_txt = jogging_path;
    config.record_name = jogging_record;
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::GetSizeScales(param& config) {
  for (int k = 0; k < 3; k++) {
    config.size_x_scales[k] = config.s_x * config.scales[k];
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::CreateHanningWindowWithCV_32F(cv::Mat* dst, int rows, int cols) {
  if (rows == 1 && cols == 1) {
    dst->at<float>(0, 0) = 1;
  } else if (rows == 1 && cols > 1) {
    float* dstData = dst->ptr<float>(0);
    for (int j = 0; j < cols; j++) {
      dstData[j] = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(j) /
                                    static_cast<double>(cols - 1)));
    }
  } else if (rows > 1 && cols == 1) {
    for (int i = 0; i < rows; i++) {
      float* dstData = dst->ptr<float>(i);
      dstData[0] = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(i) /
                                    static_cast<double>(rows - 1)));
    }
  } else {
    for (int i = 0; i < rows; i++) {
      float* dstData = dst->ptr<float>(i);
      double wr = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(i) /
                                   static_cast<double>(rows - 1)));
      for (int j = 0; j < cols; j++) {
        double wc = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(j) /
                                     static_cast<double>(cols - 1)));
        dstData[j] = static_cast<float>(wr * wc);
      }
    }
    sqrt((*dst), (*dst));
  }
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::CreateHanningWindowWithCV_64F(cv::Mat* dst, int rows, int cols) {
  if (rows == 1 && cols == 1) {
    dst->at<double>(0, 0) = 1;
  } else if (rows == 1 && cols > 1) {
    double* dstData = dst->ptr<double>(0);
    for (int j = 0; j < cols; j++) {
      dstData[j] = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(j) /
                                    static_cast<double>(cols - 1)));
    }
  } else if (rows > 1 && cols == 1) {
    for (int i = 0; i < rows; i++) {
      double* dstData = dst->ptr<double>(i);
      dstData[0] = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(i) /
                                    static_cast<double>(rows - 1)));
    }
  } else {
    for (int i = 0; i < rows; i++) {
      double* dstData = dst->ptr<double>(i);
      double wr = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(i) /
                                   static_cast<double>(rows - 1)));
      for (int j = 0; j < cols; j++) {
        double wc = 0.5 * (1.0 - cos(2.0 * CV_PI * static_cast<double>(j) /
                                     static_cast<double>(cols - 1)));
        dstData[j] = static_cast<double>(wr * wc);
      }
    }
    sqrt((*dst), (*dst));
  }
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::myCreateHanningWindow(cv::_OutputArray _dst, cv::Size winSize,
                                int type) {
  CV_Assert(type == CV_32FC1 || type == CV_64FC1);
  _dst.create(winSize, type);
  cv::Mat dst = _dst.getMat();
  int rows = dst.rows;
  int cols = dst.cols;
  if (dst.depth() == CV_32F) {
    CreateHanningWindowWithCV_32F(&dst, rows, cols);
  } else {
    CreateHanningWindowWithCV_64F(&dst, rows, cols);
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::CreMulHannWindow(cv::Size winSize, int type,
                                       cv::Mat& mulHanning) {
  int size1[2] = {1, winSize.width};
  cv::Mat selfhanning1(1, size1, CV_32FC1, cv::Scalar(0));
  myCreateHanningWindow(selfhanning1, cv::Size(1, winSize.width), CV_32FC1);
  int size2[2] = {winSize.height, 1};
  cv::Mat selfhanning2(1, size2, CV_32FC1, cv::Scalar(0));
  myCreateHanningWindow(selfhanning2, cv::Size(winSize.height, 1), CV_32FC1);
  mulHanning = selfhanning1 * selfhanning2;
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::CVMatToTensorBase(const cv::Mat& imageMat,
                                        MxBase::TensorBase& tensorBase) {
  uint32_t dataSize =
      imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU * 4;
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
                                   deviceId_);
  MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);
  APP_ERROR ret =
      MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    std::cout << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  std::vector<uint32_t> shape = {1, imageMat.channels(), imageMat.cols,
                                 imageMat.rows};
  tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                  MxBase::TENSOR_DTYPE_FLOAT32);

  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::TensorBaseToCVMat(
    const MxBase::TensorBase& tensorBase, cv::Mat& imageMat,
    const MxBase::ResizedImageInfo& resizedInfo) {
  MxBase::TensorBase tensor = tensorBase;
  int ret = tensor.ToHost();
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Tensor deploy to host failed.";
    return ret;
  }

  int outputModelWidth =
      static_cast<int>(tensor.GetShape()[MxBase::VECTOR_THIRD_INDEX]);
  int outputModelHeight =
      static_cast<int>(tensor.GetShape()[MxBase::VECTOR_FOURTH_INDEX]);

  cv::Mat imageMatInit(outputModelHeight, outputModelWidth, CV_32FC1);
  imageMat = imageMatInit;
  auto data = reinterpret_cast<float(*)[outputModelWidth][outputModelHeight]>(
      tensor.GetBuffer());
  for (size_t x = 0; x < outputModelWidth; ++x) {
    for (size_t y = 0; y < outputModelHeight; ++y) {
      imageMat.at<float>(x, y) = data[0][x][y];
    }
  }
}
APP_ERROR SiamFCBase::GetExemplar(
    const std::string& temp_video, std::vector<MxBase::TensorBase>& outputs_exemplar,
    param& config) {
  APP_ERROR ret_initpos = InitPosition(config, temp_video);
  if (ret_initpos != APP_ERR_OK) {
    LogError << "convert ones to TensorBase failed, ret="
             << GetError(ret_initpos) << ".";
    return ret_initpos;
  }
  cv::Mat src = cv::imread(config.all_files[0], cv::IMREAD_COLOR);
  cv::Mat exemplar;
  APP_ERROR ret_pretreatment =
      PreTreatMent(src, exemplar, config, 127, config.s_z);
  if (ret_pretreatment != APP_ERR_OK) {
    LogError << "PreTreatMent failed, ret=" << GetError(ret_pretreatment)
             << ".";
    return ret_pretreatment;
  }
  cv::Mat one = (cv::Mat_<float>(1, 1) << 1);
  MxBase::TensorBase exemplarBase;
  MxBase::TensorBase ones;
  APP_ERROR ret_cvchangeexemplar = CVMatToTensorBase(exemplar, exemplarBase);
  if (ret_cvchangeexemplar != APP_ERR_OK) {
    LogError << "convert exemplar to TensorBase failed , ret="
             << GetError(ret_cvchangeexemplar) << ".";
    return ret_cvchangeexemplar;
  }
  APP_ERROR ret_cvchangeone = CVMatToTensorBase(one, ones);
  if (ret_cvchangeone != APP_ERR_OK) {
    LogError << "convert ones to TensorBase failed, ret="
             << GetError(ret_cvchangeone) << ".";
    return ret_cvchangeone;
  }
  std::vector<MxBase::TensorBase> inputs;
  inputs.push_back(exemplarBase);
  inputs.push_back(ones);

  auto dtypes = model_1->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_1.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_1.outputTensors[i].tensorDims.size();
         ++j) {
      shape.push_back((uint32_t)modelDesc_1.outputTensors[i].tensorDims[j]);
    }

    MxBase::TensorBase tensor(shape, dtypes[i],
                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                              deviceId_);

    APP_ERROR ret_tensormalloc1 = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret_tensormalloc1 != APP_ERR_OK) {
      std::cout << "TensorBaseMalloc failed, ret =" << ret_tensormalloc1 << "."
           << std::endl;
      return ret_tensormalloc1;
    }

    outputs_exemplar.push_back(tensor);
  }
  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  APP_ERROR ret_infer1 =
      model_1->ModelInference(inputs, outputs_exemplar, dynamicInfo);
  if (ret_infer1 != APP_ERR_OK) {
    std::cout << "ModelInference failed, ret = " << ret_infer1 << ".";
    return ret_infer1;
  }
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::Inference(std::vector<MxBase::TensorBase>& input_exemplar,
                                MxBase::TensorBase instance,
                                std::vector<MxBase::TensorBase>& outputs_exemplar,
                                std::vector<MxBase::TensorBase>& output_exemplar) {
  input_exemplar.clear();
  input_exemplar.push_back(outputs_exemplar[0]);
  input_exemplar.push_back(instance);
  auto dtypes = model_2->GetOutputDataType();

  for (size_t i = 0; i < modelDesc_2.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_2.outputTensors[i].tensorDims.size();
         ++j) {
      shape.push_back((uint32_t)modelDesc_2.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i],
                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                              deviceId_);
    APP_ERROR ret_tensormalloc2 = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret_tensormalloc2 != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret_tensormalloc2 << ".";
      return ret_tensormalloc2;
    }
    output_exemplar.push_back(tensor);
  }
  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  APP_ERROR ret_infer2 =
      model_2->ModelInference(input_exemplar, output_exemplar, dynamicInfo);
  if (ret_infer2 != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret_infer2 << ".";
    return ret_infer2;
  }
  return APP_ERR_OK;
}

APP_ERROR SiamFCBase::GetRetInstance(
    int instance_num, param& config,
    std::vector<MxBase::TensorBase>& outputs_exemplar, cv::Mat cos_window) {
  APP_ERROR ret_getinstancesize = GetSizeScales(config);
  if (ret_getinstancesize != APP_ERR_OK) {
    std::cout << "GetSizeScales failed" << std::endl;
    return ret_getinstancesize;
  }
  cv::Mat instance_src =
      cv::imread(config.all_files[instance_num], cv::IMREAD_COLOR);
  cv::Mat exemplar_img[3];
  cv::Mat inputs_instance[3];
  cv::Mat response_mapInit[3];
  cv::Mat response_map[3];
  double response_map_max[3];
  MxBase::TensorBase instance[3];
  std::vector<MxBase::TensorBase> input_exemplar;
  MxBase::ResizedImageInfo resizedImageInfo;
  for (int n = 0; n < 3; n++) {
    std::vector<MxBase::TensorBase> output_exemplar;
    APP_ERROR ret_pretreatmentinstance = PreTreatMent(
        instance_src, exemplar_img[n], config, 255, config.size_x_scales[n]);
    if (ret_pretreatmentinstance != APP_ERR_OK) {
      std::cout << "PreTreatMent failed" << std::endl;
      return ret_pretreatmentinstance;
    }
    APP_ERROR ret_cvtotensor = CVMatToTensorBase(exemplar_img[n], instance[n]);
    if (ret_cvtotensor != APP_ERR_OK) {
      std::cout << "CVMatToTensorBase failed" << std::endl;
      return ret_cvtotensor;
    }
    APP_ERROR ret_infer3 = Inference(input_exemplar, instance[n],
                                     outputs_exemplar, output_exemplar);
    if (ret_infer3 != APP_ERR_OK) {
      std::cout << "Inference failed" << std::endl;
    }
    APP_ERROR ret_tensortocv = TensorBaseToCVMat(
        output_exemplar[0], response_mapInit[n], resizedImageInfo);
    if (ret_tensortocv != APP_ERR_OK) {
      std::cout << "TensorBaseToCVMat failed" << std::endl;
    }
  }
  double minValue = 0;
  double maxValue = 0;
  for (int n = 0; n < 3; n++) {
    cv::resize(response_mapInit[n], response_map[n], cv::Size(272, 272), 0, 0,
               cv::INTER_CUBIC);
    cv::minMaxIdx(response_map[n], &minValue, &maxValue, NULL, NULL);
    response_map_max[n] = maxValue * config.penalty[n];
  }
  int scale_index = std::max_element(response_map_max, response_map_max + 3) -
                    response_map_max;
  cv::Mat response_map_up = response_map[scale_index];
  double minValue_response = 0;
  double maxValue_response = 0;
  cv::minMaxIdx(response_map_up, &minValue_response, &maxValue_response);
  response_map_up = response_map_up - minValue_response;
  cv::Scalar sum_response = cv::sum(response_map_up);
  response_map_up = response_map_up / sum_response[0];
  response_map_up = (1 - 0.176) * response_map_up + 0.176 * cos_window;
  cv::minMaxIdx(response_map_up, &minValue_response, &maxValue_response);

  cv::Point maxLoc;
  cv::minMaxLoc(response_map_up, NULL, NULL, NULL, &maxLoc);
  double maxLoc_x = static_cast<double>(maxLoc.x);
  double maxLoc_y = static_cast<double>(maxLoc.y);
  maxLoc_x -= (271 / 2);
  maxLoc_y -= (271 / 2);
  maxLoc_x /= 2;
  maxLoc_y /= 2;

  double scale = config.scales[scale_index];
  maxLoc_x = maxLoc_x * (config.s_x * scale) / 255;
  maxLoc_y = maxLoc_y * (config.s_x * scale) / 255;
  config.target_position[0] += maxLoc_x;
  config.target_position[1] += maxLoc_y;
  std::cout << " target_position[0]: " << config.target_position[0]
       << " target_positon[1]:" << config.target_position[1] << std::endl;
  config.s_x = (0.41 + 0.59 * scale) * config.s_x;
  config.s_x = std::max(config.min_s_x, std::min(config.max_s_x, config.s_x));
  config.target_sz[0] = (0.41 + 0.59 * scale) * config.target_sz[0];
  config.target_sz[1] = (0.41 + 0.59 * scale) * config.target_sz[1];
  config.box[0] = config.target_position[0] + 1 - (config.target_sz[0]) / 2;
  config.box[1] = config.target_position[1] + 1 - (config.target_sz[1]) / 2;
  config.box[2] = config.target_sz[0];
  config.box[3] = config.target_sz[1];
  return APP_ERR_OK;
}
APP_ERROR SiamFCBase::Process(param& config) {
  std::vector<MxBase::TensorBase> outputs_exemplar;
  struct timeval start, end;
  std::ofstream outfile_record;
  std::ofstream outfile_times;
  double useTimes_ms;
  outfile_times.open(config.record_times);
  outfile_record.open(config.record_name);
  gettimeofday(&start, NULL);
  APP_ERROR ret_getfirstexemplar =
      GetExemplar(config.temp_video, outputs_exemplar, config);
  if (ret_getfirstexemplar != APP_ERR_OK) {
    std::cout << "getexemplar failed" << std::endl;
    return ret_getfirstexemplar;
  }
  gettimeofday(&end, NULL);
  useTimes_ms = end.tv_usec - start.tv_usec;
  outfile_times << useTimes_ms << std::endl;
  outfile_record << config.box[0] << "," << config.box[1] << ","
                 << config.box[2] << "," << config.box[3] << std::endl;
  cv::Mat hann;
  APP_ERROR ret_createhanningwindow =
      CreMulHannWindow(cv::Size(16 * 17, 16 * 17), CV_32FC1, hann);
  if (ret_createhanningwindow != APP_ERR_OK) {
    std::cout << "CreateMulHanningWindow failed" << std::endl;
    return ret_createhanningwindow;
  }
  cv::Scalar sum_hann = cv::sum(hann);
  cv::Mat cos_window = hann / sum_hann[0];
  for (size_t j = 1; j < config.size_s; j++) {
    gettimeofday(&start, NULL);
    GetRetInstance(j, config, outputs_exemplar, cos_window);
    gettimeofday(&end, NULL);
    useTimes_ms = end.tv_usec - start.tv_usec;
    outfile_times << useTimes_ms << std::endl;
    outfile_record << config.box[0] << "," << config.box[1] << ","
                   << config.box[2] << "," << config.box[3] << std::endl;
  }

  return APP_ERR_OK;
}

