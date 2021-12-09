import mindspore.ops as P
from mindspore import nn


class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module3(nn.Cell):
    def __init__(self):
        super(Module3, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=24,
                                 conv2d_0_out_channels=32,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module1_1 = Module1(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        return module1_1_opt


class Module8(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module1_0_conv2d_0_in_channels,
                 module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size, module1_0_conv2d_0_stride,
                 module1_0_conv2d_0_padding, module1_0_conv2d_0_pad_mode, module1_1_conv2d_0_in_channels,
                 module1_1_conv2d_0_out_channels, module1_1_conv2d_0_kernel_size, module1_1_conv2d_0_stride,
                 module1_1_conv2d_0_padding, module1_1_conv2d_0_pad_mode):
        super(Module8, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_0_conv2d_0_pad_mode)
        self.module1_1 = Module1(conv2d_0_in_channels=module1_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_1_conv2d_0_stride,
                                 conv2d_0_padding=module1_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module1_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        module1_1_opt = self.module1_1(module1_0_opt)
        opt_conv2d_0 = self.conv2d_0(module1_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_add_5 = P.Add()(opt_conv2d_4, x)
        opt_relu_6 = self.relu_6(opt_add_5)
        return opt_relu_6


class Module7(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module7, self).__init__()
        self.pad_avgpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        opt_avgpool2d_0 = self.pad_avgpool2d_0(x)
        opt_avgpool2d_0 = self.avgpool2d_0(opt_avgpool2d_0)
        opt_conv2d_1 = self.conv2d_1(opt_avgpool2d_0)
        return opt_conv2d_1


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=24,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module3_0 = Module3()
        self.pad_maxpool2d_6 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module8_0 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=256,
                                 module1_0_conv2d_0_in_channels=64,
                                 module1_0_conv2d_0_out_channels=64,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_1_conv2d_0_in_channels=64,
                                 module1_1_conv2d_0_out_channels=64,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(1, 1),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad")
        self.conv2d_8 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_14 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256)
        self.module8_1 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=512,
                                 module1_0_conv2d_0_in_channels=256,
                                 module1_0_conv2d_0_out_channels=128,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_1_conv2d_0_in_channels=128,
                                 module1_1_conv2d_0_out_channels=128,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad")
        self.module7_0 = Module7(conv2d_1_in_channels=256, conv2d_1_out_channels=512)
        self.relu_30 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512)
        self.module8_2 = Module8(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=1024,
                                 module1_0_conv2d_0_in_channels=512,
                                 module1_0_conv2d_0_out_channels=256,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_1_conv2d_0_in_channels=256,
                                 module1_1_conv2d_0_out_channels=256,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad")
        self.module7_1 = Module7(conv2d_1_in_channels=512, conv2d_1_out_channels=1024)
        self.relu_46 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module8_3 = Module8(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=2048,
                                 module1_0_conv2d_0_in_channels=1024,
                                 module1_0_conv2d_0_out_channels=512,
                                 module1_0_conv2d_0_kernel_size=(1, 1),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=0,
                                 module1_0_conv2d_0_pad_mode="valid",
                                 module1_1_conv2d_0_in_channels=512,
                                 module1_1_conv2d_0_out_channels=512,
                                 module1_1_conv2d_0_kernel_size=(3, 3),
                                 module1_1_conv2d_0_stride=(2, 2),
                                 module1_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module1_1_conv2d_0_pad_mode="pad")
        self.module7_2 = Module7(conv2d_1_in_channels=1024, conv2d_1_out_channels=2048)
        self.relu_62 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=512,
                                 conv2d_2_in_channels=512,
                                 conv2d_2_out_channels=512,
                                 conv2d_4_in_channels=512,
                                 conv2d_4_out_channels=2048)
        self.avgpool2d_70 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_71 = nn.Flatten()
        self.dense_72 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module3_0_opt = self.module3_0(opt_relu_1)
        opt_maxpool2d_6 = self.pad_maxpool2d_6(module3_0_opt)
        opt_maxpool2d_6 = self.maxpool2d_6(opt_maxpool2d_6)
        module8_0_opt = self.module8_0(opt_maxpool2d_6)
        opt_conv2d_8 = self.conv2d_8(opt_maxpool2d_6)
        opt_add_13 = P.Add()(module8_0_opt, opt_conv2d_8)
        opt_relu_14 = self.relu_14(opt_add_13)
        module0_0_opt = self.module0_0(opt_relu_14)
        module8_1_opt = self.module8_1(module0_0_opt)
        module7_0_opt = self.module7_0(module0_0_opt)
        opt_add_29 = P.Add()(module8_1_opt, module7_0_opt)
        opt_relu_30 = self.relu_30(opt_add_29)
        module0_1_opt = self.module0_1(opt_relu_30)
        module8_2_opt = self.module8_2(module0_1_opt)
        module7_1_opt = self.module7_1(module0_1_opt)
        opt_add_45 = P.Add()(module8_2_opt, module7_1_opt)
        opt_relu_46 = self.relu_46(opt_add_45)
        module0_2_opt = self.module0_2(opt_relu_46)
        module8_3_opt = self.module8_3(module0_2_opt)
        module7_2_opt = self.module7_2(module0_2_opt)
        opt_add_61 = P.Add()(module8_3_opt, module7_2_opt)
        opt_relu_62 = self.relu_62(opt_add_61)
        module0_3_opt = self.module0_3(opt_relu_62)
        opt_avgpool2d_70 = self.avgpool2d_70(module0_3_opt)
        opt_flatten_71 = self.flatten_71(opt_avgpool2d_70)
        opt_dense_72 = self.dense_72(opt_flatten_71)
        return opt_dense_72
