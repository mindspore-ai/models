/*Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  // for numpy compile warnings
#define PY_SSIZE_T_CLEAN  // essentially by https://docs.python.org/zh-cn/3/extending/extending.html
#include <Python.h>  // before any standard header files
#include <numpy/arrayobject.h>
#include <memory>
#include "include/voting_cpu.h"

namespace {
    std::unique_ptr<VotingProcess> g_processor(nullptr);
    int32_t GetArrayLength(PyArrayObject *arr, int32_t &length) {
        if (arr == nullptr) {
            LOGE("Get numpy shape failed, arr is nullptr!");
            return -1;
        }

        int32_t ndX = PyArray_NDIM(arr);
        if (ndX <= 0) {
            LOGE("Get numpy shape failed, dims is %d", ndX);
            return -1;
        }

        auto shapeX = PyArray_SHAPE(arr);
        length = 1;
        for (int32_t i = 0; i < ndX; ++i) {
            auto dimWidth = static_cast<int32_t>(shapeX[i]);
            length *= dimWidth;
        }

        return 0;
    }
}  // namespace

PyObject* PyInitVoting(PyObject* self, PyObject* args) {
    (void)self;

    int32_t classNum;
    int32_t controlPointNum;
    ModelShape voteTensorShape;
    if (!PyArg_ParseTuple(args, "iiiii",
                          &voteTensorShape.H,
                          &voteTensorShape.W,
                          &voteTensorShape.C,
                          &classNum,
                          &controlPointNum)) {
        LOGE("PyArg_ParseTuple WarpInitVoting failed!");
        return nullptr;
    }

    g_processor = std::make_unique<VotingProcess>();
    if (g_processor == nullptr) {
        return nullptr;
    }

    if (voteTensorShape.H == 0 || voteTensorShape.W == 0 || voteTensorShape.C == 0 ||
        classNum <= 0 || controlPointNum <= 0 ||
        static_cast<uint32_t>(classNum + controlPointNum * 2) != voteTensorShape.C) {
        LOGE("invalid parameters");
        return nullptr;
    }

    return Py_BuildValue("i", g_processor->Init(voteTensorShape, classNum, controlPointNum));
}

PyObject* PyVoteProcess(PyObject* self, PyObject* args) {
    (void)self;

    PyArrayObject* pyTensorData = nullptr;
    PyArrayObject* pyBox2DArr = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &pyTensorData, &pyBox2DArr)) {
        LOGE("PyArg_ParseTuple PyVoteProcess failed!");
        return nullptr;
    }

    auto outputData = static_cast<float *>(PyArray_DATA(pyTensorData));
    auto box2DArr = static_cast<float *>(PyArray_DATA(pyBox2DArr));

    int32_t npLength = 0;
    int32_t ret = GetArrayLength(pyTensorData, npLength);
    if (ret != 0) {
        LOGE("GetNumpyInfo failed!");
        return Py_BuildValue("i", -1);
    }
    std::vector<float> tensorVec(outputData, outputData + npLength);

    ret += GetArrayLength(pyBox2DArr, npLength);
    if (ret != 0) {
        LOGE("GetNumpyInfo failed!");
        return Py_BuildValue("i", -1);
    }
    std::vector<float> kpsVec(npLength, 0);

    if (!g_processor) {
        LOGE("please call function Init first");
        return Py_BuildValue("i", -1);
    }

    int classId = 0;
    ret = g_processor->PostProcess(tensorVec, classId, kpsVec);
    if (ret != 0) {
        LOGE("vote process failed!");
        return Py_BuildValue("i", -1);
    }

    memcpy(box2DArr, kpsVec.data(), kpsVec.size() * sizeof(float));

    return Py_BuildValue("i", classId);
}


/************************************ export api *************************************/
static PyMethodDef g_ransacVotingMethods[] = {
        {"init_voting", PyInitVoting,  METH_VARARGS, "some explanation"},
        {"do_voting",   PyVoteProcess, METH_VARARGS, "some explanation"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef g_ransacVotingModule = {
        PyModuleDef_HEAD_INIT,
        "libransac_voting", /* module name */
        nullptr,         /* module documentation, may be NULL */
        -1,     /* sizeof per-interpreter state of the module. -1 when module keeps state in global variables. */
        g_ransacVotingMethods,
        nullptr,
        nullptr,
        nullptr,
        nullptr
};

#define API_EXPORT  __attribute__ ((visibility ("default")))
extern "C" API_EXPORT PyObject*  PyInit_libransac_voting(void) {
    return PyModule_Create(&g_ransacVotingModule);
}
