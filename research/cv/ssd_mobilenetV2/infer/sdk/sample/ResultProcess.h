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

#pragma once

#include <queue>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
#include "boost/filesystem.hpp"
#include "boost/thread/condition_variable.hpp"
#include "boost/thread/lock_guard.hpp"
#include "boost/thread/mutex.hpp"
#include "cpprest/json.h"

class ResultProcess {
 public:
    explicit ResultProcess(std::string classes_id_path)
        : classes_id_path_(classes_id_path) {
        Init();
    }
    ResultProcess(ResultProcess &&) = delete;
    void operator()() {
        while (!is_stop) {
            if (is_finish && result.empty()) break;
            BBoxString f;
            {
                boost::unique_lock<boost::mutex> lk(m);
                cv.wait(lk, [this] {
                    return !result.empty() || (is_finish) || is_stop;
                });
                if (is_stop || (is_finish && result.empty())) {
                    lk.unlock();
                    break;
                }
                f = result.front();
                result.pop();
                lk.unlock();
            }
            web::json::value tmp = web::json::value::parse(f.res_);
            web::json::value &data = tmp["MxpiObject"];
            for (size_t i = 0; i < data.size(); i++) {
                web::json::value &d = data[i];
                float x0 = d["x0"].as_double();
                float x1 = d["x1"].as_double();
                float y0 = d["y0"].as_double();
                float y1 = d["y1"].as_double();
                web::json::value obj;
                obj["image_id"] = f.image_id_;
                web::json::value bbox =
                    web::json::value::array({x0, y0, x1 - x0, y1 - y0});
                obj["bbox"] = bbox;
                obj["score"] = d["classVec"][0]["confidence"].as_double();
                int classId =
                    static_cast<int>(d["classVec"][0]["classId"].as_integer());
                obj["category_id"] = classes_ids[classId];

                preditions.push_back(obj);
            }
            image_ids.push_back(f.image_id_);
        }

        web::json::value output = web::json::value::array(preditions);
        std::fstream out(std::string("../../data/") + "predictions.json",
                         std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            LogError << "write preditions.json error";
        }
        out << output.serialize();

        output = web::json::value::array(image_ids);
        std::fstream out1(std::string("../../data/") + "imageid.json",
                          std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            LogError << "write imageid.json error";
        }
        out1 << output.serialize();
        LogInfo << output.serialize();
    }

    void Stop() {
        is_stop = true;
        cv.notify_one();
    }
    void Finish() {
        is_finish = true;
        cv.notify_one();
    }

    void Init() {
        is_finish = false;
        is_stop = false;
        if (!boost::filesystem::is_regular_file(classes_id_path_)) {
            LogInfo << "read from : " << classes_id_path_ << " fail";
            return;
        }
        std::fstream classes_idx_stream(classes_id_path_);
        web::json::value tmp = web::json::value::parse(classes_idx_stream);
        LogInfo << "read from" << classes_id_path_ << " " << tmp;

        for (size_t i = 0; i < tmp.size(); i++) {
            classes_ids.push_back(tmp[i].as_integer());
        }
    }

    void SendResult(const std::string &res, int image_id) {
        boost::lock_guard<boost::mutex> guard(m);
        result.emplace(res, image_id);
        cv.notify_one();
    }

 private:
    class BBoxString {
     public:
        BBoxString() {}
        BBoxString(const std::string &res, int id_)
            : res_(res), image_id_(id_) {}
        std::string res_;
        int image_id_;
    };
    std::queue<BBoxString> result;
    std::vector<web::json::value> preditions;
    std::vector<web::json::value> image_ids;
    std::vector<int> classes_ids;

    boost::mutex m;
    boost::condition_variable cv;
    bool is_finish;
    bool is_stop;

    std::string classes_id_path_;
};
