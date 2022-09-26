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

#ifndef SLOWFAST_
#define SLOWFAST_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

class SLOWFAST {
 public:
  SLOWFAST(const uint32_t &deviceId, const std::string &modelPath,
           const std::string &datadir);
  ~SLOWFAST();
  void ReadImage(std::string imgPath, cv::Mat &imageMat);
  void ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
  void LoadData();
  void LoadData1();
  void LoadData2();
  void LoadData3();
  void GetData(int idx);
  std::vector<int> ReadClass();
  std::vector<std::string> ReadExcluded_keys();
  std::vector<std::vector<float>> ResizeBox(
      std::vector<std::vector<float>> boxes, int height, int width);

  std::vector<int> get_sequence(int center_idx, int half_len, int sample_rate,
                                int num_frames);
  std::vector<std::vector<float>> pad2max_float(
      std::vector<std::vector<float>> data);
  std::vector<std::vector<int>> pad2max_int(std::vector<std::vector<int>> data);
  int get_max();
  int get_batch_size();
  std::vector<cv::Mat> get_slow_pathway();
  std::vector<cv::Mat> get_fast_pathway();
  std::vector<std::vector<float>> get_padded_boxes();
  APP_ERROR Process(const std::vector<std::vector<cv::Mat>> &input1,
                    const std::vector<std::vector<cv::Mat>> &input2,
                    const std::vector<std::vector<std::vector<float>>> &input3,
                    const std::vector<float> *output);
  APP_ERROR post_process();
  void write_results(
      std::map<std::string, std::vector<std::vector<float>>> boxes,
      std::map<std::string, std::vector<float>> labels,
      std::map<std::string, std::vector<float>> scores, std::string filename);

  APP_ERROR VectorToTensorBase_mat(
      const std::vector<std::vector<cv::Mat>> &input, int idx,
      MxBase::TensorBase *tensorBase);
  APP_ERROR VectorToTensorBase_float(
      const std::vector<std::vector<std::vector<float>>> &input, int idx,
      MxBase::TensorBase *tensorBase);

  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                      std::vector<MxBase::TensorBase> *outputs);

  APP_ERROR SaveInferResult(std::vector<float> *batchFeaturePaths,
                            const std::vector<MxBase::TensorBase> &inputs);

  double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }

 private:
  std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;

  std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  double inferCostTimeMilliSec = 0.0;
  // config
  int _sample_rate = 2;
  int _video_length = 32;
  int _seq_len = _video_length * _sample_rate;
  int _num_classes = 80;
  std::vector<float> MEAN = {0.45, 0.45, 0.45};
  std::vector<float> STD = {0.225, 0.225, 0.225};
  bool BGR = false;
  bool RANDOM_FLIP = true;
  int TEST_CROP_SIZE = 224;
  bool TEST_FORCE_FLIP = false;
  int _scale_height = 256;
  int _scale_width = 384;
  int MAX_NUM_BOXES_PER_FRAME = 28;
  int frame_min = 902;
  int frame_max = 1798;

  std::string DATADIR;
  std::string ANN_DIR = "/ava/ava_annotations";
  std::string FRA_DIR = "/ava/frames";

  std::string FRAME_LIST_DIR;
  std::string TEST_LISTS = "val.csv";
  std::string FRAME_DIR;
  std::string TEST_PREDICT_BOX_LISTS =
      "person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv";
  std::string ANNOTATION_DIR;
  std::string EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv";
  std::string LABEL_MAP_FILE =
      "person_box_67091280_iou90/"
      "ava_action_list_v2.1_for_activitynet_2018.pbtxt";
  std::string GROUNDTRUTH_FILE = "ava_val_v2.2.csv";
  std::string OUTPUT_DIR = "./";
  float DETECTION_SCORE_THRESH = 0.8;

  std::string IMG_PROC_BACKEND = "cv2";
  bool REVERSE_INPUT_CHANNEL = false;
  std::string ARCH = "slowfast";
  std::string MULTI_PATHWAY_ARCH = "slowfast";
  int ALPHA = 4;

  int BATCH_SIZE = 1;
  int LOG_PERIOD = 1;
  bool FULL_TEST_ON_VAL = false;

  std::string sdk_pipeline_name = "im_slowfast";

  std::vector<std::vector<std::string>> _image_paths;
  std::vector<std::string> _video_idx_to_name;
  std::vector<std::vector<int>> _keyframe_indices;
  std::vector<std::vector<std::vector<std::vector<float>>>>
      _keyframe_boxes_and_labels;
  std::vector<cv::Mat> slow_pathway;
  std::vector<cv::Mat> fast_pathway;
  std::vector<std::vector<float>> padded_boxes;
  std::vector<int> mask;
  std::vector<std::vector<float>> padded_ori_boxes;
  std::vector<std::vector<int>> padded_metadata;

  std::vector<std::vector<float>> preds_final;
  std::vector<std::vector<float>> ori_boxes_final;
  std::vector<std::vector<int>> metadata_final;

  std::vector<std::map<int, std::vector<std::vector<float>>>> boxs_all;
};
#endif
