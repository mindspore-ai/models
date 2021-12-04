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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
typedef int row;
typedef double f64_mat_t[4][4]; /**< a matrix */
typedef double f64_vec_t[4];
#define ROW_TYPE INT
typedef int col;
#define COL_TYPE INT
typedef double cost;
#define COST_TYPE DOUBLE
#define BIG 100000
#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

/*************** DATA TYPES *******************/

typedef int boolean;
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/PostProcessBase.h"

namespace MxBase {
class Results {
 public:
  Results(uint32_t frame_id, const std::vector<cv::Mat> &online_tlwhs,
          const std::vector<int> &online_ids);
  uint32_t frame_id;
  std::vector<cv::Mat> online_tlwhs;
  std::vector<int> online_ids;
};
class TrackState {
 public:
  uint32_t New = 0;
  uint32_t Tracked = 1;
  uint32_t Lost = 2;
  uint32_t Removed = 3;
};
class BaseTrack {
 public:
  uint32_t trackId = 0;
  bool activated = false;
  uint32_t base_state;
  int next_id();

 private:
  uint32_t count = 0;
};
class KalmanFilter {
 public:
  std::map<int, float> chi2inv95 = {{1, 3.8415}, {2, 5.9915}, {3, 7.8147},
                                    {4, 9.4877}, {5, 11.070}, {6, 12.592},
                                    {7, 14.067}, {8, 15.507}, {9, 16.919}};
  uint32_t ndim;
  float dt;
  KalmanFilter();
  void chol_subtitute(cv::Mat chol_factor, cv::Mat b, f64_vec_t *f_x, int n);
  void cholesky_decomposition(const cv::Mat &A, cv::Mat &L);
  void initiate(cv::Mat measurement, cv::Mat &mean, cv::Mat &covariance);
  void multi_predict(cv::Mat &mean, std::vector<cv::Mat> &covariance);
  cv::Mat GatingDistance(cv::Mat mean, cv::Mat covariance, cv::Mat measurements,
                         bool only_position = false,
                         const std::string &metric = "maha");
  void project(cv::Mat &mean, cv::Mat &covariance);
  void update(cv::Mat &mean, cv::Mat &covariance, cv::Mat measurement);

 private:
  cv::Mat motion_mat;
  cv::Mat update_mat;
  float std_weight_position;
  float std_weight_velocity;
};

class STack : public BaseTrack {
 public:
  cv::Mat mean, covariance;
  cv::Mat smooth_feat;
  cv::Mat curr_feat;
  bool is_activated;
  KalmanFilter kalman_filter;
  float score;
  float alpha;
  uint32_t tracklet_len;
  int track_id;
  uint32_t state;
  uint32_t start_frame;
  uint32_t frame_id;
  std::vector<cv::Mat> features;
  STack();
  STack(cv::Mat tlwh, float score, cv::Mat temp_feat, uint32_t buffer_size);
  void activate(const KalmanFilter &kalman_filter, uint32_t frame_id);
  void re_activate(STack new_track, int frame_id, bool new_id = false);
  cv::Mat tlwh_to_xyah(cv::Mat tlwh);
  cv::Mat tlbr();
  cv::Mat gettlwh();
  void update(STack new_track, int frame_id, bool update_feature = true);

 private:
  cv::Mat tlwh;
  void update_features(cv::Mat temp_feat);
};
class JDETracker {
 public:
  explicit JDETracker(uint32_t frame_rate);
  std::vector<STack *> tracked_stracks;
  std::vector<STack *> lost_stracks;
  std::vector<STack *> removed_stracks;
  uint32_t frame_id = 0;
  uint32_t out_height = 0;
  uint32_t out_width = 0;
  std::vector<float> c;
  float det_thresh;
  float s = 0;
  int buffer_size;
  int max_time_lost;
  int max_per_image;
  std::string seq;
  std::string image_file;
  cv::Mat mean;
  cv::Mat std;
  KalmanFilter kalman_filter;
};
class Files {
 public:
  uint32_t frame_id = 0;
  std::vector<Results *> results;
};
class FairmotMindsporePost : public PostProcessBase {
 public:
  FairmotMindsporePost() = default;

  ~FairmotMindsporePost() = default;

  FairmotMindsporePost(const FairmotMindsporePost &other) = default;

  FairmotMindsporePost &operator=(const FairmotMindsporePost &other);

  APP_ERROR Init(
      const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

  APP_ERROR DeInit() override;

  APP_ERROR Process(const std::vector<TensorBase> &tensors,
                    MxBase::JDETracker &tracker, MxBase::Files &file);

  bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

 private:
  std::vector<STack *> ObjectDetectionOutput(
      const std::vector<TensorBase> &tensors, MxBase::JDETracker &tracker);
  void TransformPreds(const cv::Mat &coords, MxBase::JDETracker tracker,
                      cv::Mat &target_coords);
  void PostProcess(cv::Mat &det, const MxBase::JDETracker &tracker);
  void TensorBaseToCVMat(cv::Mat &imageMat, const MxBase::TensorBase &tensor);
  void FuseMotion(MxBase::KalmanFilter &kalman_filter, cv::Mat &cost_matrix,
                  std::vector<STack *> tracks, std::vector<STack *> detections,
                  bool only_position = false, float lambda_ = 0.98);
  std::vector<cv::Mat> LinearAssignment(cv::Mat cost_matrix, float thresh);
  void lap(cost **assigncost, col *rowsol, row *colsol, cost *u, cost *v,
           int row, int col);
  std::vector<cv::Mat> get_lap(cost **assigncost, col *rowsol, row *colsol,
                               cost *u, cost *v, int row, int col, int dim);
  std::vector<STack *> JointStracks(std::vector<STack *> tlista,
                                    std::vector<STack *> tlistb);
  std::vector<STack *> SubStracks(std::vector<STack *> tlista,
                                  std::vector<STack *> tlistb);
  void RemoveDuplicateStracks(std::vector<STack *> &stracksa,
                              std::vector<STack *> &stracksb);
  cv::Mat IouDistance(std::vector<STack *> atracks,
                      std::vector<STack *> btracks);
  cv::Mat BboxOverlaps(std::vector<cv::Mat> boxes,
                       std::vector<cv::Mat> query_boxes);
  std::vector<STack *> Get_output_stracks(
      MxBase::JDETracker &tracker,
      const std::vector<STack *> &activated_starcks,
      const std::vector<STack *> &refind_stracks,
      std::vector<STack *> lost_stracks, std::vector<STack *> removed_stracks);
  void Get_detections(cv::Mat det, std::vector<STack *> &detections,
                      cv::Mat id_feature);
  void Get_dists(cv::Mat &dists, std::vector<STack *> &detections,
                 std::vector<STack *> &strack_pool);
  void Update_Starcks(const std::vector<STack *> &strack_pool,
                      std::vector<STack *> &detections, const cv::Mat &matches,
                      std::vector<STack *> &activated_starcks,
                      std::vector<STack *> &refind_stracks,
                      const MxBase::JDETracker &tracker);
  void get_result(int *collist, int cols, cost *d, cost min, int rows,
                  int &endofpath, row *colsol, cost **assigncost, cost *v,
                  int *pred);
  void func(int &numfree, int *free, cost **assigncost, row *colsol, cost *v,
            col *rowsol, int cols);
  void MultiPredict(std::vector<STack *> &stracks);
  const uint32_t DEFAULT_CLASS_NUM_MS = 80;
  const float DEFAULT_SCORE_THRESH_MS = 0.7;
  const float DEFAULT_IOU_THRESH_MS = 0.5;
  const uint32_t DEFAULT_RPN_MAX_NUM_MS = 1000;
  const uint32_t DEFAULT_MAX_PER_IMG_MS = 128;

  uint32_t classNum_ = DEFAULT_CLASS_NUM_MS;
  float scoreThresh_ = DEFAULT_SCORE_THRESH_MS;
  float iouThresh_ = DEFAULT_IOU_THRESH_MS;
  uint32_t rpnMaxNum_ = DEFAULT_RPN_MAX_NUM_MS;
  uint32_t maxPerImg_ = DEFAULT_MAX_PER_IMG_MS;
};

extern "C" {
std::shared_ptr<MxBase::FairmotMindsporePost> GetObjectInstance();
}
}  // namespace MxBase
