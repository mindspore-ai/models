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

#include <fstream>
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "MxBase/Log/Log.h"
#include "MxBase/PostProcessBases/PostProcessBase.h"
#include "boost/filesystem.hpp"
#include "cpprest/json.h"
#include "infer/mxbase/CommandFlagParser.h"
#include "infer/mxbase/FunctionTimer.h"
#include "infer/mxbase/MxBaseInfer.h"
#include "infer/mxbase/MxImage.h"
#include "infer/mxbase/MxUtil.h"
#include "infer/mxbase/SSDInfer.h"
#include "infer/mxbase/SSDPostProcessor.h"

DEFINE_string(model, "../data/models/ssd-mobilenet-v2.om", "om model path.");
DEFINE_bool(debug, false, "is using debug");
DEFINE_string(out_images, "../data/", "out image file or path.");
DEFINE_int32(device, 0, "device id used");
DEFINE_string(color, "bgr", "input color space need for model.");
DEFINE_string(classes_path, "../data/classes.json", "classes path");
DEFINE_string(classes_id_path, "../data/classes_id.json", "classes id path");
DEFINE_string(coco_path, "../data/coco2017/", "coco data dir");
DEFINE_int32(width, 320, "model_width");
DEFINE_int32(height, 320, "model_height");
DEFINE_string(config_cfg_path, "../data/config.cfg", "cfg path");
DEFINE_string(label_path, "../data/coco_ssd_mobile_net_v2.name", "label_path");

static bool g_loop_stop = false;

void softINT(int signo) {
    LogInfo << "user ctr-c quit loop!\n";
    g_loop_stop = true;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, softINT);

    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = google::GLOG_INFO;
    OptionManager::getInstance()->parseCommandLineFlags(argc, argv);

    LogInfo << "model : " << FLAGS_model;
    LogInfo << "debug : " << FLAGS_debug;
    LogInfo << "out_images : " << FLAGS_out_images;
    LogInfo << "device : " << FLAGS_device;
    LogInfo << "color : " << FLAGS_color;
    LogInfo << "classes_path : " << FLAGS_classes_path;
    LogInfo << "classes_id_path : " << FLAGS_classes_id_path;
    LogInfo << "coco_path : " << FLAGS_coco_path;

    sdk_infer::mxbase_infer::SSDInfer infer(
        FLAGS_device, FLAGS_classes_path, FLAGS_classes_id_path,
        FLAGS_out_images, FLAGS_debug, FLAGS_width,
        FLAGS_height, FLAGS_config_cfg_path, FLAGS_label_path);
    if (!infer.Init(FLAGS_model)) {
        LogError << "init inference module failed !!!";
        return -1;
    }

    // load coco data
    std::fstream coco_stream(FLAGS_coco_path +
                             "annotations/instances_val2017.json");
    if (!coco_stream.is_open()) {
        LogError << "read file from : "
                 << FLAGS_coco_path + "annotations/instances_val2017.json"
                 << " fail";
        return -1;
    }
    web::json::value coco_json = web::json::value::parse(coco_stream);
    web::json::array images_info = coco_json["images"].as_array();
    web::json::array annotations = coco_json["annotations"].as_array();

    std::map<int, int> image_category_id;
    std::map<int, int> image_iscrowd;

    for (auto &ann_info : annotations) {
        int iscrowd = ann_info["iscrowd"].as_integer();
        int category_id = ann_info["category_id"].as_integer();
        int id = ann_info["image_id"].as_integer();
        image_category_id[id] = category_id;
        image_iscrowd[id] = iscrowd;
    }

    sdk_infer::mxbase_infer::FunctionTimer timer;
    sdk_infer::mxbase_infer::FunctionStats func_stat("", "us");

    int num = 0;
    for (auto &image_info : images_info) {
        if (g_loop_stop) break;

        timer.start_timer();
        std::string image_name =
            FLAGS_coco_path + "val2017/" + image_info["file_name"].as_string();

        int id = image_info["id"].as_integer();

        if (image_iscrowd[id] == 1 || image_category_id[id] >= 91) {
            LogDebug << "process " << image_name
                     << " ............................................no ok";
            continue;
        }

        MxBase::TensorBase input;
        if (!infer.LoadImageToModel(image_name, &input,
                                    image_info["id"].as_integer())) {
            LogError << "load image:" << image_name
                     << " to device input[0] error!!!";
            break;
        }
        std::vector<MxBase::TensorBase> inputs{input};
        std::vector<MxBase::TensorBase> outputs;
        bool result = infer.Inference(&inputs, &outputs);

        timer.calculate_time();

        func_stat.update_time(timer.get_elapsed_time_in_microseconds());
        if (!result) break;
        num++;
    }
    infer.SaveResult();
    infer.UnInit();
}
