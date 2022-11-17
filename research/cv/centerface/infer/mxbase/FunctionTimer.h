/*
 * Copyright (c) 2021.Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef OFFICIAL_CV_CENTERFACE_INFER_MXBASE_FUNCTIONTIMER_H_
#define OFFICIAL_CV_CENTERFACE_INFER_MXBASE_FUNCTIONTIMER_H_

#include <sys/time.h>
#include <string>
#include <iostream>
#include <limits>

class FunctionStats {
 private:
    int64_t longest_time = 0;  // the longest function running time
    int64_t shortest_time =
        std::numeric_limits<int64_t>::max();  // the shortest function running
                                              // time
    int64_t total_time = 0;
    int64_t count = 0;  // number of times the function being called
    std::string function_name;
    std::string time_unit;

 public:
    FunctionStats(const std::string &function_name_ = "",
                  const std::string &time_unit_ = "ms")
        : function_name(function_name_), time_unit(time_unit_) {}

    void update_time(const int64_t& running_time) {
        ++count;
        total_time += running_time;

        if (running_time > longest_time) {
            longest_time = running_time;
        }
        if (running_time < shortest_time) {
            shortest_time = running_time;
        }
    }

    void print_stats() const {
        std::cout << "========================================================="
                  << std::endl;
        std::cout << "Function name: " << function_name << std::endl;
        if (count == 0) {
            std::cout << "this function hasn't been run yet" << std::endl;
        } else {
            std::cout << "this function has run " << count << " times"
                      << std::endl;
            std::cout << "average function running time: "
                      << (int64_t)(total_time / count) << time_unit
                      << std::endl;
            std::cout << "longest function running time: " << longest_time
                      << time_unit << std::endl;
            std::cout << "shortest function running time: " << shortest_time
                      << time_unit << std::endl;
        }
        std::cout << "========================================================="
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

#endif  // OFFICIAL_CV_CENTERFACE_INFER_MXBASE_FUNCTIONTIMER_H_
