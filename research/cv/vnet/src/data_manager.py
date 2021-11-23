# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""convert CT images to numpy data"""
import os
import numpy as np
import SimpleITK as sitk


class DataManager():
    """MR image manager module"""

    def __init__(self, imagelist, data_folder, parameters):
        self.imagelist = imagelist
        self.imageFolder = os.path.join(data_folder, 'img')
        self.GTFolder = os.path.join(data_folder, 'gt')
        self.params = parameters

    def createImageFileList(self):
        self.imageFileList = np.genfromtxt(self.imagelist, dtype=str)
        self.imageFileList.sort()

    def createGTFileList(self):
        self.GTFileList = [f.split('.')[0]+'_segmentation.mhd' for f in self.imageFileList]
        self.GTFileList.sort()

    def loadImages(self):
        self.sitkImages = dict()
        rescalFilt = sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)
        for f in self.imageFileList:
            img_id = f.split('.')[0]
            self.sitkImages[img_id] = rescalFilt.Execute(sitk.Cast(
                                sitk.ReadImage(os.path.join(self.imageFolder, f)), sitk.sitkFloat32))

    def loadGT(self):
        self.sitkGT = dict()
        for f in self.GTFileList:
            img_id = f.split('.')[0]
            self.sitkGT[img_id] = sitk.Cast(sitk.ReadImage(os.path.join(self.GTFolder, f)) > 0.5, sitk.sitkFloat32)

    def loadTrainingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadTestingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadInferData(self):
        self.createImageFileList()
        self.loadImages()

    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages, sitk.sitkLinear)
        for key in dat:
            mean = np.mean(dat[key][dat[key] > 0])
            std = np.std(dat[key][dat[key] > 0])
            dat[key] -= mean
            dat[key] /= std
        return dat

    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT, sitk.sitkLinear)
        for key in dat:
            dat[key] = (dat[key] > 0.5).astype(dtype=np.float32)
        return dat

    def getNumpyData(self, dat, method):
        """get numpy data from MR images"""

        ret = dict()
        for key in dat:
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize'][1],
                                 self.params['VolSize'][2]], dtype=np.float32)
            img = dat[key]
            factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]
            factorSize = np.asarray(img.GetSize() * factor, dtype=float)
            newSize = np.max([factorSize, self.params['VolSize']], axis=0)
            newSize = newSize.astype(dtype='int')
            T = sitk.AffineTransform(3)
            T.SetMatrix(img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
            resampler.SetSize(newSize.tolist())
            resampler.SetInterpolator(method)
            if self.params['normDir']:
                resampler.SetTransform(T.GetInverse())
            imgResampled = resampler.Execute(img)
            imgCentroid = np.asarray(newSize, dtype=float) / 2.0
            imgStartPx = (imgCentroid - np.array(self.params['VolSize']) / 2.0).astype(dtype='int')
            regionExtractor = sitk.RegionOfInterestImageFilter()
            size_2_set = np.array(self.params['VolSize']).astype(dtype='int')
            regionExtractor.SetSize(size_2_set.tolist())
            regionExtractor.SetIndex(imgStartPx.tolist())
            imgResampledCropped = regionExtractor.Execute(imgResampled)
            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])
        return ret

    def writeResultsFromNumpyLabel(self, result, key, resultTag, ext):
        """get MR images from numpy data"""

        img = self.sitkImages[key]
        resultDir = self.params['dirPredictionImage']
        if not os.path.exists(resultDir):
            os.makedirs(resultDir, exist_ok=True)
        print("original img shape{}".format(img.GetSize()))
        toWrite = sitk.Image(img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], sitk.sitkFloat32)
        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                 self.params['dstRes'][2]]
        factorSize = np.asarray(img.GetSize() * factor, dtype=float)
        newSize = np.max([factorSize, np.array(self.params['VolSize'])], axis=0)
        newSize = newSize.astype(dtype=int)
        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize.tolist())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())
        toWrite = resampler.Execute(toWrite)
        imgCentroid = np.asarray(newSize, dtype=float) / 2.0
        imgStartPx = (imgCentroid - np.array(self.params['VolSize']) / 2.0).astype(dtype=int)
        for dstX, srcX in zip(range(0, result.shape[0]),
                              range(imgStartPx[0], int(imgStartPx[0] + self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]),
                                  range(imgStartPx[1], int(imgStartPx[1]+self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]),
                                      range(imgStartPx[2], int(imgStartPx[2]+self.params['VolSize'][2]))):
                    toWrite.SetPixel(int(srcX), int(srcY), int(srcZ), float(result[dstX, dstY, dstZ]))
        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())
        if self.params['normDir']:
            resampler.SetTransform(T)
        toWrite = resampler.Execute(toWrite)
        thfilter = sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(toWrite)
        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite, sitk.sitkUInt8))
        arrCC = np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [2, 1, 0])
        lab = np.zeros(int(np.max(arrCC) + 1), dtype=float)
        for i in range(1, int(np.max(arrCC) + 1)):
            lab[i] = np.sum(arrCC == i)
        activeLab = np.argmax(lab)
        toWrite = (toWritecc == activeLab)
        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(resultDir, key + resultTag + ext))
        writer.Execute(toWrite)
