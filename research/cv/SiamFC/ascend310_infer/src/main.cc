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
#include <gflags/gflags.h>
#include <opencv2/imgproc/types_c.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "inc/utils.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/types.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "include/dataset/vision_ascend.h"

using mindspore::Context;
using mindspore::DataType;
using mindspore::Graph;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::Model;
using mindspore::ModelType;
using mindspore::MSTensor;
using mindspore::Serialization;
using mindspore::Status;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::transforms::TypeCast;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Resize;
using namespace cv;
using namespace std;

DEFINE_string(model_path1, "/home/siamfc/model1.mindir", "model path");
DEFINE_string(model_path2, "/home/siamfc/model2_change.mindir", "model path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision_mode, "allow_fp32_to_fp16", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(aipp_path, "./aipp.cfg", "aipp path");
DEFINE_string(device_target, "Ascend310", "device target");
DEFINE_string(code_path, "/home/Siamfc/", "code path");
DEFINE_string(seq_root_path, "/home/siamfc/OTB2013/", "OTB route");
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

struct param {
  const int none = 1;
  const int* one = &none;
  size_t s_one = 4;
  size_t size_s;
  double init_x;
  double init_y;
  double init_w;
  double init_h;
  double target_position[2];
  double target_sz[2];
  double wc_z;
  double hc_z;
  double s_z;
  double scale_z;
  double penalty[3] = {0.9745, 1, 0.9745};
  double scales[3] = {0.96385542, 1.00, 1.0375};
  string dataset_path_txt;
  string record_name;
  string record_times;
  double s_x;
  double min_s_x;
  double max_s_x;
  double size_x_scales[3];
  vector<double> box;
  vector<string> all_files;
};
Mat hwc2chw(Mat dst, size_t resize_detection) {
  std::vector<float> dst_data;
  std::vector<cv::Mat> bgrChannels(3);
  cv::split(dst, bgrChannels);
  for (size_t i = 0; i < bgrChannels.size(); i++) {
    std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
    dst_data.insert(dst_data.end(), data.begin(), data.end());
  }
  cv::Mat srcMat;
  srcMat = cv::Mat(dst_data, true);
  cv::Mat dst_img = srcMat.reshape(3, resize_detection);
  return dst_img;
}
void pretreatment(cv::Mat src, cv::Mat& target, param config, int size,
                  double s_x) {
  cv::Mat cropImg = crop_and_pad(src, config.target_position[0],
                                 config.target_position[1], size, s_x);
  cv::Mat exemplar_FLOAT;
  cropImg.convertTo(exemplar_FLOAT, CV_32FC3);
  target = hwc2chw(exemplar_FLOAT, size);
}
void init_position(param& config, string& temp_video) {
  config.all_files = GetAllFiles(FLAGS_seq_root_path, temp_video);
  config.box = Getpos(config.dataset_path_txt);
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
}

void getPath(param& config, string& temp_video, int jogging_count) {
  config.dataset_path_txt =
      FLAGS_seq_root_path + "/" + temp_video + "/" + "groundtruth_rect.txt";
  config.record_name =
      FLAGS_code_path + "/results/OTB2013/SiamFC/" + temp_video + ".txt";
  config.record_times = FLAGS_code_path + "/results/OTB2013/SiamFC/times/" +
                        temp_video + "_time.txt";
  if (temp_video == "Jogging") {
    auto jogging_path = FLAGS_seq_root_path + "/" + temp_video + "/" +
                        "groundtruth_rect" + "." +
                        std::to_string(jogging_count) + ".txt";
    auto jogging_record = FLAGS_code_path + "/results/OTB2013/SiamFC/" +
                          temp_video + "." + std::to_string(jogging_count) +
                          ".txt";
    config.dataset_path_txt = jogging_path;
    config.record_name = jogging_record;
  }
}

void getSizeScales(param& config) {
  for (int k = 0; k < 3; k++) {
    config.size_x_scales[k] = config.s_x * config.scales[k];
  }
}

void getExemplar(string& temp_video, vector<MSTensor>& outputs_exemplar,
                 vector<MSTensor>& inputs_exemplar, Model& model1,
                 param& config, int jogging_count) {
  getPath(config, temp_video, jogging_count);
  std::vector<MSTensor> model_inputs = model1.GetInputs();
  init_position(config, temp_video);
  cv::Mat src = cv::imread(config.all_files[0], cv::IMREAD_COLOR);
  cv::Mat exemplar;
  pretreatment(src, exemplar, config, 127, config.s_z);
  cout << "box :" << config.box[0] << " " << config.box[1] << " "
       << config.box[2] << " " << config.box[3] << endl;
  size_t size_buffer = exemplar.size().width * exemplar.size().height * 4 * 3;
  mindspore::MSTensor image("x", mindspore::DataType::kNumberTypeFloat32,
                            {static_cast<int64_t>(3), static_cast<int64_t>(127),
                             static_cast<int64_t>(127)},
                            exemplar.data, size_buffer);
  std::vector<int64_t> shape = image.Shape();
  inputs_exemplar.clear();
  inputs_exemplar.emplace_back(
      model_inputs[0].Name(), model_inputs[0].DataType(),
      model_inputs[0].Shape(), image.Data().get(), image.DataSize());
  inputs_exemplar.emplace_back(
      model_inputs[1].Name(), model_inputs[1].DataType(),
      model_inputs[1].Shape(), config.one, config.s_one);
  Status ret_instance;
  ret_instance =
      model1.Predict(inputs_exemplar, &outputs_exemplar);  // get exemplar img
  if (ret_instance != kSuccess) {
    cout << " Failed predict" << endl;
  } else {
    cout << " Success predict" << endl;
  }
}
void preInstance(vector<MSTensor>& input_exemplar,
                 vector<MSTensor>& outputs_exemplar,
                 vector<MSTensor>& output_exemplar,
                 vector<MSTensor>& model_inputs_instance, Model& model2,
                 MSTensor& instance) {
  input_exemplar.clear();
  input_exemplar.emplace_back(
      model_inputs_instance[0].Name(), model_inputs_instance[0].DataType(),
      model_inputs_instance[0].Shape(), outputs_exemplar[0].Data().get(),
      outputs_exemplar[0].DataSize());
  input_exemplar.emplace_back(model_inputs_instance[1].Name(),
                              model_inputs_instance[1].DataType(),
                              model_inputs_instance[1].Shape(),
                              instance.Data().get(), instance.DataSize());
  model2.Predict(input_exemplar, &output_exemplar);
}

void getRetInstance(int instance_num, vector<MSTensor>& inputs,
                    vector<MSTensor>& outputs,
                    vector<MSTensor>& outputs_exemplar, Mat cos_window,
                    param& config, Model& model2) {
  getSizeScales(config);
  vector<MSTensor> model_inputs_instance = model2.GetInputs();
  cv::Mat instance_src;
  instance_src = cv::imread(config.all_files[instance_num], cv::IMREAD_COLOR);
  cv::Mat exemplar_img[3];
  cv::Mat inputs_instance[3];
  cv::Mat response_mapInit[3];
  cv::Mat response_map[3];
  double response_map_max[3];
  std::vector<MSTensor> input_exemplar;
  std::vector<MSTensor> output_exemplar1;
  std::vector<MSTensor> output_exemplar2;
  std::vector<MSTensor> output_exemplar3;
  for (int n = 0; n < 3; n++) {
    pretreatment(instance_src, exemplar_img[n], config, 255,
                 config.size_x_scales[n]);
  }
  size_t size_buffer_instance =
      exemplar_img[0].size().width * exemplar_img[0].size().height * 3 * 4;
  mindspore::MSTensor instance1(
      "y", mindspore::DataType::kNumberTypeFloat32,
      {static_cast<int64_t>(3), static_cast<int64_t>(255),
       static_cast<int64_t>(255)},
      exemplar_img[0].data, size_buffer_instance);
  mindspore::MSTensor instance2(
      "y", mindspore::DataType::kNumberTypeFloat32,
      {static_cast<int64_t>(3), static_cast<int64_t>(255),
       static_cast<int64_t>(255)},
      exemplar_img[1].data, size_buffer_instance);
  mindspore::MSTensor instance3(
      "y", mindspore::DataType::kNumberTypeFloat32,
      {static_cast<int64_t>(3), static_cast<int64_t>(255),
       static_cast<int64_t>(255)},
      exemplar_img[2].data, size_buffer_instance);

  preInstance(input_exemplar, outputs_exemplar, output_exemplar1,
              model_inputs_instance, model2, instance1);
  preInstance(input_exemplar, outputs_exemplar, output_exemplar2,
              model_inputs_instance, model2, instance2);
  preInstance(input_exemplar, outputs_exemplar, output_exemplar3,
              model_inputs_instance, model2, instance3);
  response_mapInit[0] =
      cv::Mat(17, 17, CV_32FC1, output_exemplar1[0].MutableData());
  response_mapInit[1] =
      cv::Mat(17, 17, CV_32FC1, output_exemplar2[0].MutableData());
  response_mapInit[2] =
      cv::Mat(17, 17, CV_32FC1, output_exemplar3[0].MutableData());

  double minValue = 0;
  double maxValue = 0;
  for (int n = 0; n < 3; n++) {
    cv::resize(response_mapInit[n], response_map[n], Size(272, 272), 0, 0,
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
  Scalar sum_response = sum(response_map_up);
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
  cout << " target_position[0]: " << config.target_position[0]
       << " target_positon[1]:" << config.target_position[1] << endl;
  config.s_x = (0.41 + 0.59 * scale) * config.s_x;
  config.s_x = max(config.min_s_x, min(config.max_s_x, config.s_x));
  config.target_sz[0] = (0.41 + 0.59 * scale) * config.target_sz[0];
  config.target_sz[1] = (0.41 + 0.59 * scale) * config.target_sz[1];
  config.box[0] = config.target_position[0] + 1 - (config.target_sz[0]) / 2;
  config.box[1] = config.target_position[1] + 1 - (config.target_sz[1]) / 2;
  config.box[2] = config.target_sz[0];
  config.box[3] = config.target_sz[1];
}

void myCreateHanningWindow(OutputArray _dst, cv::Size winSize, int type) {
  CV_Assert(type == CV_32FC1 || type == CV_64FC1);
  _dst.create(winSize, type);
  Mat dst = _dst.getMat();
  int rows = dst.rows;
  int cols = dst.cols;
  if (dst.depth() == CV_32F) {
    if (rows == 1 && cols == 1) {
      dst.at<float>(0, 0) = 1;
    } else if (rows == 1 && cols > 1) {
      float* dstData = dst.ptr<float>(0);
      for (int j = 0; j < cols; j++) {
        dstData[j] =
            0.5 * (1.0 - cos(2.0 * CV_PI * (double)j / (double)(cols - 1)));
      }
    } else if (rows > 1 && cols == 1) {
      for (int i = 0; i < rows; i++) {
        float* dstData = dst.ptr<float>(i);
        dstData[0] =
            0.5 * (1.0 - cos(2.0 * CV_PI * (double)i / (double)(rows - 1)));
      }

    } else {
      for (int i = 0; i < rows; i++) {
        float* dstData = dst.ptr<float>(i);
        double wr =
            0.5 * (1.0 - cos(2.0 * CV_PI * (double)i / (double)(rows - 1)));
        for (int j = 0; j < cols; j++) {
          double wc =
              0.5 * (1.0 - cos(2.0 * CV_PI * (double)j / (double)(cols - 1)));
          dstData[j] = (float)(wr * wc);
        }
      }
      sqrt(dst, dst);
    }
  } else {
    if (rows == 1 && cols == 1) {
      dst.at<double>(0, 0) = 1;
    } else if (rows == 1 && cols > 1) {
      double* dstData = dst.ptr<double>(0);
      for (int j = 0; j < cols; j++) {
        dstData[j] =
            0.5 * (1.0 - cos(2.0 * CV_PI * (double)j / (double)(cols - 1)));
      }
    } else if (rows > 1 && cols == 1) {
      for (int i = 0; i < rows; i++) {
        double* dstData = dst.ptr<double>(i);
        dstData[0] =
            0.5 * (1.0 - cos(2.0 * CV_PI * (double)i / (double)(rows - 1)));
      }
    } else {
      for (int i = 0; i < rows; i++) {
        double* dstData = dst.ptr<double>(i);
        double wr =
            0.5 * (1.0 - cos(2.0 * CV_PI * (double)i / (double)(rows - 1)));
        for (int j = 0; j < cols; j++) {
          double wc =
              0.5 * (1.0 - cos(2.0 * CV_PI * (double)j / (double)(cols - 1)));
          dstData[j] = (double)(wr * wc);
        }
      }
      sqrt(dst, dst);
    }
  }
}
Mat createMulHanningWindow(cv::Size winSize, int type) {
  int size1[2] = {1, winSize.width};
  cv::Mat selfhanning1(1, size1, CV_32FC1, cv::Scalar(0));
  myCreateHanningWindow(selfhanning1, cv::Size(1, winSize.width), CV_32FC1);
  int size2[2] = {winSize.height, 1};
  cv::Mat selfhanning2(1, size2, CV_32FC1, cv::Scalar(0));
  myCreateHanningWindow(selfhanning2, cv::Size(winSize.height, 1), CV_32FC1);
  cv::Mat mulHanning;
  mulHanning = selfhanning1 * selfhanning2;
  return mulHanning;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_model_path1).empty()) {
    std::cout << "Invalid model" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310_info);
  // load  graph1
  Graph graph1;
  Status ret =
      Serialization::Load(FLAGS_model_path1, ModelType::kMindIR, &graph1);
  cout << "Load model success" << endl;
  if (ret != kSuccess) {
    std::cout << "Load model failed." << std::endl;
    return 1;
  }
  Model model1;
  Status ret_build = model1.Build(GraphCell(graph1), context);
  if (ret_build != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  } else {
    cout << "  Build success  " << endl;
  }
  // load graph2
  Graph graph2;
  Status ret_graph2 =
      Serialization::Load(FLAGS_model_path2, ModelType::kMindIR, &graph2);
  if (ret_graph2 != kSuccess) {
    cout << " load graph2 failed" << endl;
  } else {
    cout << " load graph2 Success" << endl;
  }
  Model model2;
  Status ret_build2 = model2.Build(GraphCell(graph2), context);
  if (ret_build2 != kSuccess) {
    cout << " build graph2 failed" << endl;
  } else {
    cout << " build graph2 Success" << endl;
  }

  auto all_files = GetAllFiles(FLAGS_seq_root_path, all_videos[0]);
  if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }
  int jogging_count = 1;
  std::map<double, double> costTime_map;
  size_t size_v = all_videos.size();
  for (size_t i = 0; i < size_v; ++i) {
    param config;
    vector<MSTensor> inputs_exemplar;
    vector<MSTensor> outputs_exemplar;
    struct timeval start, end;
    double startTime_ms, endTime_ms, useTime_ms;
    gettimeofday(&start, NULL);
    getExemplar(all_videos[i], outputs_exemplar, inputs_exemplar, model1,
                config, jogging_count);
    cout << "record:" << config.record_name << "  " << config.record_times
         << endl;
    gettimeofday(&end, NULL);
    costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    ofstream outfile_record;
    ofstream outfile_times;
    outfile_times.open(config.record_times);
    outfile_record.open(config.record_name);
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    useTime_ms = endTime_ms - startTime_ms;
    outfile_times << useTime_ms << std::endl;
    outfile_record << config.box[0] << "," << config.box[1] << ","
                   << config.box[2] << "," << config.box[3] << endl;
    cv::Mat hann;
    hann = createMulHanningWindow(cv::Size(16 * 17, 16 * 17), CV_32FC1);
    Scalar sum_hann = sum(hann);
    cv::Mat cos_window = hann / sum_hann[0];  // create hanning
    // load graph2
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    for (size_t j = 1; j < config.size_s; j++) {
      gettimeofday(&start, NULL);
      getRetInstance(j, inputs, outputs, outputs_exemplar, cos_window, config,
                     model2);
      gettimeofday(&end, NULL);
      costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
      startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
      endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 100;
      useTime_ms = endTime_ms - startTime_ms;
      outfile_times << useTime_ms << std::endl;
      outfile_record << config.box[0] << "," << config.box[1] << ","
                     << config.box[2] << "," << config.box[3] << endl;
    }
    if (all_videos[i] == "Jogging" && jogging_count == 1) {
      i--;
      jogging_count++;
    }
  }
  double average = 0.0;
  int infer_cnt = 0;
  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = 0.0;
    diff = iter->second - iter->first;
    average += diff;
    infer_cnt++;
  }

  average = average / infer_cnt;

  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average
           << " ms of infer_count " << infer_cnt << std::endl;
  std::cout << "NN inference cost average time: " << average
            << "ms of infer_count " << infer_cnt << std::endl;
  std::string file_name =
      "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
  file_stream << timeCost.str();
  file_stream.close();
  costTime_map.clear();
  return 0;
  cout << "End project" << endl;
  return 0;
}
