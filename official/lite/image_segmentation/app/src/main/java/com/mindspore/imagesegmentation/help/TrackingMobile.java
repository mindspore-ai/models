/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.imagesegmentation.help;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.core.graphics.ColorUtils;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.CpuBindMode;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;


import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashSet;
import java.util.List;

public class TrackingMobile {
    private static final String TAG = "TrackingMobile";

    private static final String IMAGESEGMENTATIONMODEL = "segment_model.ms";
    private static final int imageSize = 257;
    public static final int NUM_CLASSES = 21;
    private static final float IMAGE_MEAN = 127.5F;
    private static final float IMAGE_STD = 127.5F;

    public static final int[] segmentColors = new int[2];

    private Bitmap maskBitmap;
    private Bitmap resultBitmap;
    private HashSet itemsFound = new HashSet();

    private final Context mContext;

    private Model model;

    public TrackingMobile(Context context) {
        mContext = context;
        init();
    }

    private MappedByteBuffer loadModel(Context context, String modelName) {
        FileInputStream fis = null;
        AssetFileDescriptor fileDescriptor = null;

        try {
            fileDescriptor = context.getAssets().openFd(modelName);
            fis = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = fis.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLen = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLen);
        } catch (IOException var24) {
            Log.e("MS_LITE", "Load model failed");
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException var23) {
                    Log.e("MS_LITE", "Close file failed");
                }
            }

            if (fileDescriptor != null) {
                try {
                    fileDescriptor.close();
                } catch (IOException var22) {
                    Log.e("MS_LITE", "Close fileDescriptor failed");
                }
            }

        }

        return null;
    }

    public void init() {
        // Load the .ms model.
        model = new Model();

        // Create and init config.
        MSContext context = new MSContext();
        if (!context.init(2, CpuBindMode.MID_CPU, false)) {
            Log.e(TAG, "Init context failed");
            return;
        }
        if (!context.addDeviceInfo(DeviceType.DT_CPU, false, 0)) {
            Log.e(TAG, "Add device info failed");
            return;
        }
        MappedByteBuffer modelBuffer = loadModel(mContext, IMAGESEGMENTATIONMODEL);
        if(modelBuffer == null) {
            Log.e(TAG, "Load model failed");
            return;
        }
        // Create the MindSpore lite session.
        boolean ret = model.build(modelBuffer, ModelType.MT_MINDIR,context);
        if(!ret) {
            Log.e(TAG, "Build model failed");
        }
    }

    public ModelTrackingResult execute(Bitmap bitmap) {
        // Set input tensor values.
        List<MSTensor> inputs = model.getInputs();
        if (inputs.size() != 1) {
            Log.e(TAG, "inputs.size() != 1");
            return null;
        }

        float resource_height = bitmap.getHeight();
        float resource_weight = bitmap.getWidth();

        Bitmap scaledBitmap = BitmapUtils.scaleBitmapAndKeepRatio(bitmap, imageSize, imageSize);
        ByteBuffer contentArray = BitmapUtils.bitmapToByteBuffer(scaledBitmap, imageSize, imageSize, IMAGE_MEAN, IMAGE_STD);

        MSTensor inTensor = inputs.get(0);
        inTensor.setData(contentArray);

        // Run graph to infer results.
        if (!model.predict()) {
            Log.e(TAG, "Run graph failed");
            return null;
        }

        // Get output tensor values.
        List<MSTensor> outputs = model.getOutputs();
        for (MSTensor output : outputs) {
            if (output == null) {
                Log.e(TAG, "Output is null");
                return null;
            }
            float[] results = output.getFloatData();
            float[] result = new float[output.elementsNum()];

            int batch = output.getShape()[0];
            int channel = output.getShape()[1];
            int weight = output.getShape()[2];
            int height = output.getShape()[3];
            int plane = weight * height;

            for (int n = 0; n < batch; n++) {
                for (int c = 0; c < channel; c++) {
                    for (int hw = 0; hw < plane; hw++) {
                        result[n * channel * plane + hw * channel + c] = results[n * channel * plane + c * plane + hw];
                    }
                }
            }
            ByteBuffer bytebuffer_results = floatArrayToByteArray(result);

            convertBytebufferMaskToBitmap(
                    bytebuffer_results, imageSize, imageSize, scaledBitmap,
                    segmentColors
            );
            //scaledBitmap resize成resource_height，resource_weight
            scaledBitmap = BitmapUtils.scaleBitmapAndKeepRatio(scaledBitmap, (int) resource_height, (int) resource_weight);
            resultBitmap = BitmapUtils.scaleBitmapAndKeepRatio(resultBitmap, (int) resource_height, (int) resource_weight);
            maskBitmap = BitmapUtils.scaleBitmapAndKeepRatio(maskBitmap, (int) resource_height, (int) resource_weight);
        }
        return new ModelTrackingResult(resultBitmap, scaledBitmap, maskBitmap, this.formatExecutionLog(), itemsFound);
    }

    private static ByteBuffer floatArrayToByteArray(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer;
    }

    private void convertBytebufferMaskToBitmap(ByteBuffer inputBuffer, int imageWidth,
                                               int imageHeight, Bitmap backgroundImage, int[] colors) {
        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf);
        resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf);
        Bitmap scaledBackgroundImage =
                BitmapUtils.scaleBitmapAndKeepRatio(backgroundImage, imageWidth, imageHeight);
        int[][] mSegmentBits = new int[imageWidth][imageHeight];
        inputBuffer.rewind();
        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                float maxVal = 0f;
                mSegmentBits[x][y] = 0;
                for (int i = 0; i < NUM_CLASSES; i++) {
                    float value = inputBuffer.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + i) * 4);
                    if (i == 0 || value > maxVal) {
                        maxVal = value;
                        if (i == 15) {
                            mSegmentBits[x][y] = i;
                        } else {
                            mSegmentBits[x][y] = 0;
                        }
                    }
                }
                itemsFound.add(mSegmentBits[x][y]);

                int newPixelColor = ColorUtils.compositeColors(
                        colors[mSegmentBits[x][y] == 0 ? 0 : 1],
                        scaledBackgroundImage.getPixel(x, y)
                );
                resultBitmap.setPixel(x, y, newPixelColor);
                maskBitmap.setPixel(x, y, mSegmentBits[x][y] == 0 ? colors[0] : scaledBackgroundImage.getPixel(x, y));
            }
        }
    }

    // Note: we must release the memory at the end, otherwise it will cause the memory leak.
    public void free() {
        model.free();
    }


    private final String formatExecutionLog() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input Image Size: " + imageSize * imageSize + '\n');
        return sb.toString();
    }

}
