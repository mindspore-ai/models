import mindspore.ops as P
from mindspore import nn


class Module3(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module3, self).__init__()
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


class Module6(nn.Cell):
    def __init__(self):
        super(Module6, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=32,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module3_1 = Module3(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module3_1_opt = self.module3_1(module3_0_opt)
        return module3_1_opt


class Module10(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module3_0_conv2d_0_in_channels,
                 module3_0_conv2d_0_out_channels, module3_0_conv2d_0_kernel_size, module3_0_conv2d_0_stride,
                 module3_0_conv2d_0_padding, module3_0_conv2d_0_pad_mode, module3_1_conv2d_0_in_channels,
                 module3_1_conv2d_0_out_channels, module3_1_conv2d_0_kernel_size, module3_1_conv2d_0_stride,
                 module3_1_conv2d_0_padding, module3_1_conv2d_0_pad_mode):
        super(Module10, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_0_conv2d_0_stride,
                                 conv2d_0_padding=module3_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_0_conv2d_0_pad_mode)
        self.module3_1 = Module3(conv2d_0_in_channels=module3_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_1_conv2d_0_stride,
                                 conv2d_0_padding=module3_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_1_conv2d_0_pad_mode)
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
        module3_0_opt = self.module3_0(x)
        module3_1_opt = self.module3_1(module3_0_opt)
        opt_conv2d_0 = self.conv2d_0(module3_1_opt)
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


class Module14(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module14, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module13(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module13, self).__init__()
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


class Module31(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_4_in_channels, module0_3_conv2d_4_out_channels,
                 module0_4_conv2d_0_in_channels, module0_4_conv2d_0_out_channels, module0_4_conv2d_2_in_channels,
                 module0_4_conv2d_2_out_channels, module0_4_conv2d_4_in_channels, module0_4_conv2d_4_out_channels,
                 module0_5_conv2d_0_in_channels, module0_5_conv2d_0_out_channels, module0_5_conv2d_2_in_channels,
                 module0_5_conv2d_2_out_channels, module0_5_conv2d_4_in_channels, module0_5_conv2d_4_out_channels,
                 module0_6_conv2d_0_in_channels, module0_6_conv2d_0_out_channels, module0_6_conv2d_2_in_channels,
                 module0_6_conv2d_2_out_channels, module0_6_conv2d_4_in_channels, module0_6_conv2d_4_out_channels,
                 module0_7_conv2d_0_in_channels, module0_7_conv2d_0_out_channels, module0_7_conv2d_2_in_channels,
                 module0_7_conv2d_2_out_channels, module0_7_conv2d_4_in_channels, module0_7_conv2d_4_out_channels):
        super(Module31, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels)
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_4_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_4_conv2d_4_out_channels)
        self.module0_5 = Module0(conv2d_0_in_channels=module0_5_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_5_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_5_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_5_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_5_conv2d_4_out_channels)
        self.module0_6 = Module0(conv2d_0_in_channels=module0_6_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_6_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_6_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_6_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_6_conv2d_4_out_channels)
        self.module0_7 = Module0(conv2d_0_in_channels=module0_7_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_7_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_7_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_7_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_7_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        return module0_7_opt


class Module30(nn.Cell):
    def __init__(self):
        super(Module30, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module0_1 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module0_2 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module0_3 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module6_0 = Module6()
        self.pad_maxpool2d_6 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module10_0 = Module10(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   module3_0_conv2d_0_in_channels=64,
                                   module3_0_conv2d_0_out_channels=64,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=64,
                                   module3_1_conv2d_0_out_channels=64,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
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
        self.module14_0 = Module14(module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=64,
                                   module0_0_conv2d_2_in_channels=64,
                                   module0_0_conv2d_2_out_channels=64,
                                   module0_0_conv2d_4_in_channels=64,
                                   module0_0_conv2d_4_out_channels=256,
                                   module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=64,
                                   module0_1_conv2d_2_in_channels=64,
                                   module0_1_conv2d_2_out_channels=64,
                                   module0_1_conv2d_4_in_channels=64,
                                   module0_1_conv2d_4_out_channels=256)
        self.module10_1 = Module10(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   module3_0_conv2d_0_in_channels=256,
                                   module3_0_conv2d_0_out_channels=128,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=128,
                                   module3_1_conv2d_0_out_channels=128,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.module13_0 = Module13(conv2d_1_in_channels=256, conv2d_1_out_channels=512)
        self.relu_37 = nn.ReLU()
        self.module14_1 = Module14(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_0_conv2d_2_out_channels=128,
                                   module0_0_conv2d_4_in_channels=128,
                                   module0_0_conv2d_4_out_channels=512,
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_2_in_channels=128,
                                   module0_1_conv2d_2_out_channels=128,
                                   module0_1_conv2d_4_in_channels=128,
                                   module0_1_conv2d_4_out_channels=512)
        self.module0_0 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512)
        self.module10_2 = Module10(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   module3_0_conv2d_0_in_channels=512,
                                   module3_0_conv2d_0_out_channels=256,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=256,
                                   module3_1_conv2d_0_out_channels=256,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.module13_1 = Module13(conv2d_1_in_channels=512, conv2d_1_out_channels=1024)
        self.relu_67 = nn.ReLU()
        self.module31_0 = Module31(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024)
        self.module31_1 = Module31(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024)
        self.module30_0 = Module30()
        self.module14_2 = Module14(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024)
        self.module10_3 = Module10(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=2048,
                                   module3_0_conv2d_0_in_channels=1024,
                                   module3_0_conv2d_0_out_channels=512,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=512,
                                   module3_1_conv2d_0_out_channels=512,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.module13_2 = Module13(conv2d_1_in_channels=1024, conv2d_1_out_channels=2048)
        self.relu_230 = nn.ReLU()
        self.module14_3 = Module14(module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=2048,
                                   module0_1_conv2d_0_in_channels=2048,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_2_in_channels=512,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_conv2d_4_in_channels=512,
                                   module0_1_conv2d_4_out_channels=2048)
        self.avgpool2d_245 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_246 = nn.Flatten()
        self.dense_247 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module6_0_opt = self.module6_0(opt_relu_1)
        opt_maxpool2d_6 = self.pad_maxpool2d_6(module6_0_opt)
        opt_maxpool2d_6 = self.maxpool2d_6(opt_maxpool2d_6)
        module10_0_opt = self.module10_0(opt_maxpool2d_6)
        opt_conv2d_8 = self.conv2d_8(opt_maxpool2d_6)
        opt_add_13 = P.Add()(module10_0_opt, opt_conv2d_8)
        opt_relu_14 = self.relu_14(opt_add_13)
        module14_0_opt = self.module14_0(opt_relu_14)
        module10_1_opt = self.module10_1(module14_0_opt)
        module13_0_opt = self.module13_0(module14_0_opt)
        opt_add_36 = P.Add()(module10_1_opt, module13_0_opt)
        opt_relu_37 = self.relu_37(opt_add_36)
        module14_1_opt = self.module14_1(opt_relu_37)
        module0_0_opt = self.module0_0(module14_1_opt)
        module10_2_opt = self.module10_2(module0_0_opt)
        module13_1_opt = self.module13_1(module0_0_opt)
        opt_add_66 = P.Add()(module10_2_opt, module13_1_opt)
        opt_relu_67 = self.relu_67(opt_add_66)
        module31_0_opt = self.module31_0(opt_relu_67)
        module31_1_opt = self.module31_1(module31_0_opt)
        module30_0_opt = self.module30_0(module31_1_opt)
        module14_2_opt = self.module14_2(module30_0_opt)
        module10_3_opt = self.module10_3(module14_2_opt)
        module13_2_opt = self.module13_2(module14_2_opt)
        opt_add_229 = P.Add()(module10_3_opt, module13_2_opt)
        opt_relu_230 = self.relu_230(opt_add_229)
        module14_3_opt = self.module14_3(opt_relu_230)
        opt_avgpool2d_245 = self.avgpool2d_245(module14_3_opt)
        opt_flatten_246 = self.flatten_246(opt_avgpool2d_245)
        opt_dense_247 = self.dense_247(opt_flatten_246)
        return opt_dense_247
