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
#ifndef OFFICIAL_RECOMMEND_WIDE_AND_DEEP_INFER_MXBASE_FUNCTIONTIMER_H_
#define OFFICIAL_RECOMMEND_WIDE_AND_DEEP_INFER_MXBASE_FUNCTIONTIMER_H_

#include <sys/time.h>

#include <iostream>
#include <limits>
#include <string>

#include "MxBase/Log/Log.h"

class FunctionStats {
 private:
    int64_t longest_time = 0;  // the longest function running time
    int64_t shortest_time =
        std::numeric_limits<int>::max();  // the shortest function running time
    int64_t total_time = 0;
    int64_t count = 0;  // number of times the function being called
    std::string function_name;
    int64_t proportion_to_sec;  // the proportion of current time unit over
                                // second, the default is 1e3 (for millisecond)

 public:
    explicit FunctionStats(const std::string &function_name_ = "",
                           const uint64_t &proportion = 1e3)
        : function_name(function_name_), proportion_to_sec(proportion) {}

    void update_time(const int64_t &running_time) {
        ++count;
        total_time += running_time;

        if (running_time > longest_time) longest_time = running_time;
        if (running_time < shortest_time) shortest_time = running_time;
    }

    void print_stats(const std::string &time_unit = "ms") const {
        LogInfo << "========================================================="
                << std::endl;
        LogInfo << "Function name: " << function_name << std::endl;
        LogInfo << "this function has run " << count << " times" << std::endl;
        LogInfo << "average function running time: "
                << (int64_t)(total_time / count) << time_unit << std::endl;
        LogInfo << "longest function running time: " << longest_time
                << time_unit << std::endl;
        LogInfo << "shortest function running time: " << shortest_time
                << time_unit << std::endl;
        LogInfo << "fps: "
                << count / (static_cast<double>(total_time) / proportion_to_sec)
                << std::endl;
        LogInfo << "========================================================="
                << std::endl;
    }
};

// time the function for running once, default unit is millisecond
class FunctionTimer {
 private:
    int64_t start = 0;
    int64_t end = 0;

 public:
    void start_timer() {
        timeval cur_time;
        gettimeofday(&cur_time, NULL);
        start = cur_time.tv_sec * 1e6 + cur_time.tv_usec;
    }

    void calculate_time() {
        timeval cur_time;
        gettimeofday(&cur_time, NULL);
        end = cur_time.tv_sec * 1e6 + cur_time.tv_usec;
    }

    int64_t get_elapsed_time_in_milliseconds() const {
        return (end - start) / 1000;
    }

    int64_t get_elapsed_time_in_microseconds() const { return end - start; }
};

#endif  // OFFICIAL_RECOMMEND_WIDE_AND_DEEP_INFER_MXBASE_FUNCTIONTIMER_H_
