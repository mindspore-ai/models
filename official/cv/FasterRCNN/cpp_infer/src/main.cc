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
#include <sys/time.h>

#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");
DEFINE_int32(IMAGEWIDTH, 1280, "image width");
DEFINE_int32(IMAGEHEIGHT, 768, "image height");
DEFINE_bool(KEEP_RATIO, true, "keep_ratio");
DEFINE_bool(RESTOREBBOX, true, "restore bbox");

int RescaleImage(const MSTensor &input, MSTensor *output) {
  std::shared_ptr<TensorTransform> normalize(new Normalize({103.53, 116.28, 123.675}, {57.375, 57.120, 58.395}));
  Execute composeNormalize({normalize});
  std::vector<int64_t> shape = input.Shape();
  auto imgResize = MSTensor();
  auto imgPad = MSTensor();

  float widthScale, heightScale;
  widthScale = static_cast<float>(FLAGS_IMAGEWIDTH) / shape[1];
  heightScale = static_cast<float>(FLAGS_IMAGEHEIGHT) / shape[0];
  Status ret;
  if (widthScale < heightScale) {
    int heightSize = shape[0] * widthScale;
    std::shared_ptr<TensorTransform> resize(new Resize({heightSize, FLAGS_IMAGEWIDTH}));
    Execute composeResizeWidth({resize});
    ret = composeResizeWidth(input, &imgResize);
    if (ret != kSuccess) {
      std::cout << "ERROR: Resize Width failed." << std::endl;
      return 1;
    }

    int paddingSize = FLAGS_IMAGEHEIGHT - heightSize;
    std::shared_ptr<TensorTransform> pad(new Pad({0, 0, 0, paddingSize}));
    Execute composePad({pad});
    ret = composePad(imgResize, &imgPad);
    if (ret != kSuccess) {
      std::cout << "ERROR: Height Pad failed." << std::endl;
      return 1;
    }

    ret = composeNormalize(imgPad, output);
    if (ret != kSuccess) {
      std::cout << "ERROR: Normalize failed." << std::endl;
      return 1;
    }
  } else {
    int widthSize = shape[1] * heightScale;
    std::shared_ptr<TensorTransform> resize(new Resize({FLAGS_IMAGEHEIGHT, widthSize}));
    Execute composeResizeHeight({resize});
    ret = composeResizeHeight(input, &imgResize);
    if (ret != kSuccess) {
      std::cout << "ERROR: Resize Height failed." << std::endl;
      return 1;
    }

    int paddingSize = FLAGS_IMAGEWIDTH - widthSize;
    std::shared_ptr<TensorTransform> pad(new Pad({0, 0, paddingSize, 0}));
    Execute composePad({pad});
    ret = composePad(imgResize, &imgPad);
    if (ret != kSuccess) {
      std::cout << "ERROR: Width Pad failed." << std::endl;
      return 1;
    }

    ret = composeNormalize(imgPad, output);
    if (ret != kSuccess) {
      std::cout << "ERROR: Normalize failed." << std::endl;
      return 1;
    }
  }
  return 0;
}

int ResizeImage(const MSTensor &input, MSTensor *output) {
  std::shared_ptr<TensorTransform> normalize(new Normalize({123.675, 116.28, 103.53}, {58.395, 57.120, 57.375}));
  Execute composeNormalize({normalize});
  auto imgResize = MSTensor();

  Status ret;

  std::shared_ptr<TensorTransform> resize(new Resize({FLAGS_IMAGEHEIGHT, FLAGS_IMAGEWIDTH}));
  Execute composeResize({resize});
  ret = composeResize(input, &imgResize);
  if (ret != kSuccess) {
    std::cout << "ERROR: Resize failed." << std::endl;
    return 1;
  }
  ret = composeNormalize(imgResize, output);
  if (ret != kSuccess) {
    std::cout << "ERROR: Normalize failed." << std::endl;
    return 1;
  }
  return 0;
}

void statistic_results(const std::map<double, double> &costTime_map) {
  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: " << average << "ms of infer_count " << inferCount << std::endl;

  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
}

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, ascend310, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Status ret;
  std::vector<MSTensor> model_inputs = model.GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  std::map<double, double> costTime_map;
  size_t size = all_files.size();
  std::shared_ptr<TensorTransform> decode(new Decode());
  Execute composeDecode({decode});
  std::shared_ptr<TensorTransform> hwc2chw(new HWC2CHW());
  Execute composeTranspose({hwc2chw});

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << all_files[i] << std::endl;
    auto imgDecode = MSTensor();
    auto image = ReadFileToTensor(all_files[i]);
    ret = composeDecode(image, &imgDecode);
    if (ret != kSuccess) {
      std::cout << "ERROR: Decode failed." << std::endl;
      return 1;
    }
    auto imgRescale = MSTensor();
    if (FLAGS_KEEP_RATIO) {
      RescaleImage(imgDecode, &imgRescale);
    } else {
      ResizeImage(imgDecode, &imgRescale);
    }
    auto img = MSTensor();
    composeTranspose(imgRescale, &img);

    std::vector<int64_t> shape = imgDecode.Shape();

    float widthScale = static_cast<float>(FLAGS_IMAGEWIDTH) / shape[1];
    float heightScale = static_cast<float>(FLAGS_IMAGEHEIGHT) / shape[0];

    float imgInfo[4];
    imgInfo[0] = shape[0];
    imgInfo[1] = shape[1];
    size_t heightIndex = 2;
    size_t widthIndex = 3;
    if (FLAGS_KEEP_RATIO) {
      float resizeScale = widthScale < heightScale ? widthScale : heightScale;
      imgInfo[heightIndex] = resizeScale;
      imgInfo[widthIndex] = resizeScale;
    } else {
      imgInfo[heightIndex] = heightScale;
      imgInfo[widthIndex] = widthScale;
    }

    MSTensor imgMeta("imgMeta", DataType::kNumberTypeFloat32, {static_cast<int64_t>(4)}, imgInfo, 16);
    bool restore_bbox = static_cast<bool>(FLAGS_RESTOREBBOX);
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(), img.Data().get(),
                        img.DataSize());
    if (restore_bbox) {
      inputs.emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(), model_inputs[1].Shape(),
                          imgMeta.Data().get(), imgMeta.DataSize());
    }

    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << all_files[i] << " failed." << std::endl;
      return 1;
    }
    size_t uSecScale = 1000000;
    size_t msScale = 1000;
    startTimeMs = (1.0 * start.tv_sec * uSecScale + start.tv_usec) / msScale;
    endTimeMs = (1.0 * end.tv_sec * uSecScale + end.tv_usec) / msScale;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(all_files[i], outputs);
  }
  statistic_results(costTime_map);
  costTime_map.clear();
  return 0;
}
