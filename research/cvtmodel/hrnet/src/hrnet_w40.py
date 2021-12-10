import mindspore.ops as P
from mindspore import nn


class Module5(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module5, self).__init__()
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


class Module15(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, module5_0_conv2d_0_in_channels, module5_0_conv2d_0_out_channels,
                 module5_0_conv2d_0_kernel_size, module5_0_conv2d_0_stride, module5_0_conv2d_0_padding,
                 module5_0_conv2d_0_pad_mode, module5_1_conv2d_0_in_channels, module5_1_conv2d_0_out_channels,
                 module5_1_conv2d_0_kernel_size, module5_1_conv2d_0_stride, module5_1_conv2d_0_padding,
                 module5_1_conv2d_0_pad_mode):
        super(Module15, self).__init__()
        self.module5_0 = Module5(conv2d_0_in_channels=module5_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module5_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module5_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module5_0_conv2d_0_stride,
                                 conv2d_0_padding=module5_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module5_0_conv2d_0_pad_mode)
        self.module5_1 = Module5(conv2d_0_in_channels=module5_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module5_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module5_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module5_1_conv2d_0_stride,
                                 conv2d_0_padding=module5_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module5_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module5_0_opt = self.module5_0(x)
        module5_1_opt = self.module5_1(module5_0_opt)
        opt_conv2d_0 = self.conv2d_0(module5_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_5_in_channels, conv2d_5_out_channels, conv2d_7_in_channels, conv2d_7_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
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
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(in_channels=conv2d_5_in_channels,
                                  out_channels=conv2d_5_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()
        self.conv2d_7 = nn.Conv2d(in_channels=conv2d_7_in_channels,
                                  out_channels=conv2d_7_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_9 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_add_3 = P.Add()(opt_conv2d_2, x)
        opt_relu_4 = self.relu_4(opt_add_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_conv2d_7 = self.conv2d_7(opt_relu_6)
        opt_add_8 = P.Add()(opt_conv2d_7, opt_relu_4)
        opt_relu_9 = self.relu_9(opt_add_8)
        return opt_relu_9


class Module16(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels,
                 module0_0_conv2d_7_in_channels, module0_0_conv2d_7_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_5_in_channels, module0_1_conv2d_5_out_channels, module0_1_conv2d_7_in_channels,
                 module0_1_conv2d_7_out_channels):
        super(Module16, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_5_in_channels=module0_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_0_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_5_in_channels=module0_1_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_1_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_1_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_1_conv2d_7_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module7(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, resizenearestneighbor_1_size):
        super(Module7, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.resizenearestneighbor_1 = P.ResizeNearestNeighbor(size=resizenearestneighbor_1_size, align_corners=False)

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_resizenearestneighbor_1 = self.resizenearestneighbor_1(opt_conv2d_0)
        return opt_resizenearestneighbor_1


class Module11(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module5_0_conv2d_0_in_channels,
                 module5_0_conv2d_0_out_channels, module5_0_conv2d_0_kernel_size, module5_0_conv2d_0_stride,
                 module5_0_conv2d_0_padding, module5_0_conv2d_0_pad_mode):
        super(Module11, self).__init__()
        self.module5_0 = Module5(conv2d_0_in_channels=module5_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module5_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module5_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module5_0_conv2d_0_stride,
                                 conv2d_0_padding=module5_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module5_0_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module5_0_opt = self.module5_0(x)
        opt_conv2d_0 = self.conv2d_0(module5_0_opt)
        return opt_conv2d_0


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module5_0 = Module5(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module15_0 = Module15(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=64,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=64,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.conv2d_5 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_11 = nn.ReLU()
        self.module15_1 = Module15(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=256,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=64,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.relu_18 = nn.ReLU()
        self.module15_2 = Module15(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=256,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=64,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.relu_25 = nn.ReLU()
        self.module15_3 = Module15(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=256,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=64,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.relu_32 = nn.ReLU()
        self.module5_1 = Module5(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=40,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module5_2 = Module5(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=80,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module16_0 = Module16(module0_0_conv2d_0_in_channels=40,
                                   module0_0_conv2d_0_out_channels=40,
                                   module0_0_conv2d_2_in_channels=40,
                                   module0_0_conv2d_2_out_channels=40,
                                   module0_0_conv2d_5_in_channels=40,
                                   module0_0_conv2d_5_out_channels=40,
                                   module0_0_conv2d_7_in_channels=40,
                                   module0_0_conv2d_7_out_channels=40,
                                   module0_1_conv2d_0_in_channels=40,
                                   module0_1_conv2d_0_out_channels=40,
                                   module0_1_conv2d_2_in_channels=40,
                                   module0_1_conv2d_2_out_channels=40,
                                   module0_1_conv2d_5_in_channels=40,
                                   module0_1_conv2d_5_out_channels=40,
                                   module0_1_conv2d_7_in_channels=40,
                                   module0_1_conv2d_7_out_channels=40)
        self.module16_1 = Module16(module0_0_conv2d_0_in_channels=80,
                                   module0_0_conv2d_0_out_channels=80,
                                   module0_0_conv2d_2_in_channels=80,
                                   module0_0_conv2d_2_out_channels=80,
                                   module0_0_conv2d_5_in_channels=80,
                                   module0_0_conv2d_5_out_channels=80,
                                   module0_0_conv2d_7_in_channels=80,
                                   module0_0_conv2d_7_out_channels=80,
                                   module0_1_conv2d_0_in_channels=80,
                                   module0_1_conv2d_0_out_channels=80,
                                   module0_1_conv2d_2_in_channels=80,
                                   module0_1_conv2d_2_out_channels=80,
                                   module0_1_conv2d_5_in_channels=80,
                                   module0_1_conv2d_5_out_channels=80,
                                   module0_1_conv2d_7_in_channels=80,
                                   module0_1_conv2d_7_out_channels=80)
        self.module7_0 = Module7(conv2d_0_in_channels=80,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_85 = nn.ReLU()
        self.conv2d_77 = nn.Conv2d(in_channels=40,
                                   out_channels=80,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_81 = nn.ReLU()
        self.module5_3 = Module5(conv2d_0_in_channels=80,
                                 conv2d_0_out_channels=160,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module16_2 = Module16(module0_0_conv2d_0_in_channels=40,
                                   module0_0_conv2d_0_out_channels=40,
                                   module0_0_conv2d_2_in_channels=40,
                                   module0_0_conv2d_2_out_channels=40,
                                   module0_0_conv2d_5_in_channels=40,
                                   module0_0_conv2d_5_out_channels=40,
                                   module0_0_conv2d_7_in_channels=40,
                                   module0_0_conv2d_7_out_channels=40,
                                   module0_1_conv2d_0_in_channels=40,
                                   module0_1_conv2d_0_out_channels=40,
                                   module0_1_conv2d_2_in_channels=40,
                                   module0_1_conv2d_2_out_channels=40,
                                   module0_1_conv2d_5_in_channels=40,
                                   module0_1_conv2d_5_out_channels=40,
                                   module0_1_conv2d_7_in_channels=40,
                                   module0_1_conv2d_7_out_channels=40)
        self.module16_3 = Module16(module0_0_conv2d_0_in_channels=80,
                                   module0_0_conv2d_0_out_channels=80,
                                   module0_0_conv2d_2_in_channels=80,
                                   module0_0_conv2d_2_out_channels=80,
                                   module0_0_conv2d_5_in_channels=80,
                                   module0_0_conv2d_5_out_channels=80,
                                   module0_0_conv2d_7_in_channels=80,
                                   module0_0_conv2d_7_out_channels=80,
                                   module0_1_conv2d_0_in_channels=80,
                                   module0_1_conv2d_0_out_channels=80,
                                   module0_1_conv2d_2_in_channels=80,
                                   module0_1_conv2d_2_out_channels=80,
                                   module0_1_conv2d_5_in_channels=80,
                                   module0_1_conv2d_5_out_channels=80,
                                   module0_1_conv2d_7_in_channels=80,
                                   module0_1_conv2d_7_out_channels=80)
        self.module16_4 = Module16(module0_0_conv2d_0_in_channels=160,
                                   module0_0_conv2d_0_out_channels=160,
                                   module0_0_conv2d_2_in_channels=160,
                                   module0_0_conv2d_2_out_channels=160,
                                   module0_0_conv2d_5_in_channels=160,
                                   module0_0_conv2d_5_out_channels=160,
                                   module0_0_conv2d_7_in_channels=160,
                                   module0_0_conv2d_7_out_channels=160,
                                   module0_1_conv2d_0_in_channels=160,
                                   module0_1_conv2d_0_out_channels=160,
                                   module0_1_conv2d_2_in_channels=160,
                                   module0_1_conv2d_2_out_channels=160,
                                   module0_1_conv2d_5_in_channels=160,
                                   module0_1_conv2d_5_out_channels=160,
                                   module0_1_conv2d_7_in_channels=160,
                                   module0_1_conv2d_7_out_channels=160)
        self.module7_1 = Module7(conv2d_0_in_channels=80,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.module7_2 = Module7(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_162 = nn.ReLU()
        self.conv2d_149 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_3 = Module7(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=80,
                                 resizenearestneighbor_1_size=(28, 28))
        self.relu_163 = nn.ReLU()
        self.module11_0 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_145 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_167 = nn.ReLU()
        self.module16_5 = Module16(module0_0_conv2d_0_in_channels=40,
                                   module0_0_conv2d_0_out_channels=40,
                                   module0_0_conv2d_2_in_channels=40,
                                   module0_0_conv2d_2_out_channels=40,
                                   module0_0_conv2d_5_in_channels=40,
                                   module0_0_conv2d_5_out_channels=40,
                                   module0_0_conv2d_7_in_channels=40,
                                   module0_0_conv2d_7_out_channels=40,
                                   module0_1_conv2d_0_in_channels=40,
                                   module0_1_conv2d_0_out_channels=40,
                                   module0_1_conv2d_2_in_channels=40,
                                   module0_1_conv2d_2_out_channels=40,
                                   module0_1_conv2d_5_in_channels=40,
                                   module0_1_conv2d_5_out_channels=40,
                                   module0_1_conv2d_7_in_channels=40,
                                   module0_1_conv2d_7_out_channels=40)
        self.module16_6 = Module16(module0_0_conv2d_0_in_channels=80,
                                   module0_0_conv2d_0_out_channels=80,
                                   module0_0_conv2d_2_in_channels=80,
                                   module0_0_conv2d_2_out_channels=80,
                                   module0_0_conv2d_5_in_channels=80,
                                   module0_0_conv2d_5_out_channels=80,
                                   module0_0_conv2d_7_in_channels=80,
                                   module0_0_conv2d_7_out_channels=80,
                                   module0_1_conv2d_0_in_channels=80,
                                   module0_1_conv2d_0_out_channels=80,
                                   module0_1_conv2d_2_in_channels=80,
                                   module0_1_conv2d_2_out_channels=80,
                                   module0_1_conv2d_5_in_channels=80,
                                   module0_1_conv2d_5_out_channels=80,
                                   module0_1_conv2d_7_in_channels=80,
                                   module0_1_conv2d_7_out_channels=80)
        self.module16_7 = Module16(module0_0_conv2d_0_in_channels=160,
                                   module0_0_conv2d_0_out_channels=160,
                                   module0_0_conv2d_2_in_channels=160,
                                   module0_0_conv2d_2_out_channels=160,
                                   module0_0_conv2d_5_in_channels=160,
                                   module0_0_conv2d_5_out_channels=160,
                                   module0_0_conv2d_7_in_channels=160,
                                   module0_0_conv2d_7_out_channels=160,
                                   module0_1_conv2d_0_in_channels=160,
                                   module0_1_conv2d_0_out_channels=160,
                                   module0_1_conv2d_2_in_channels=160,
                                   module0_1_conv2d_2_out_channels=160,
                                   module0_1_conv2d_5_in_channels=160,
                                   module0_1_conv2d_5_out_channels=160,
                                   module0_1_conv2d_7_in_channels=160,
                                   module0_1_conv2d_7_out_channels=160)
        self.module7_4 = Module7(conv2d_0_in_channels=80,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.module7_5 = Module7(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_243 = nn.ReLU()
        self.conv2d_225 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_6 = Module7(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=80,
                                 resizenearestneighbor_1_size=(28, 28))
        self.relu_244 = nn.ReLU()
        self.module11_1 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_228 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_245 = nn.ReLU()
        self.module16_8 = Module16(module0_0_conv2d_0_in_channels=40,
                                   module0_0_conv2d_0_out_channels=40,
                                   module0_0_conv2d_2_in_channels=40,
                                   module0_0_conv2d_2_out_channels=40,
                                   module0_0_conv2d_5_in_channels=40,
                                   module0_0_conv2d_5_out_channels=40,
                                   module0_0_conv2d_7_in_channels=40,
                                   module0_0_conv2d_7_out_channels=40,
                                   module0_1_conv2d_0_in_channels=40,
                                   module0_1_conv2d_0_out_channels=40,
                                   module0_1_conv2d_2_in_channels=40,
                                   module0_1_conv2d_2_out_channels=40,
                                   module0_1_conv2d_5_in_channels=40,
                                   module0_1_conv2d_5_out_channels=40,
                                   module0_1_conv2d_7_in_channels=40,
                                   module0_1_conv2d_7_out_channels=40)
        self.module16_9 = Module16(module0_0_conv2d_0_in_channels=80,
                                   module0_0_conv2d_0_out_channels=80,
                                   module0_0_conv2d_2_in_channels=80,
                                   module0_0_conv2d_2_out_channels=80,
                                   module0_0_conv2d_5_in_channels=80,
                                   module0_0_conv2d_5_out_channels=80,
                                   module0_0_conv2d_7_in_channels=80,
                                   module0_0_conv2d_7_out_channels=80,
                                   module0_1_conv2d_0_in_channels=80,
                                   module0_1_conv2d_0_out_channels=80,
                                   module0_1_conv2d_2_in_channels=80,
                                   module0_1_conv2d_2_out_channels=80,
                                   module0_1_conv2d_5_in_channels=80,
                                   module0_1_conv2d_5_out_channels=80,
                                   module0_1_conv2d_7_in_channels=80,
                                   module0_1_conv2d_7_out_channels=80)
        self.module16_10 = Module16(module0_0_conv2d_0_in_channels=160,
                                    module0_0_conv2d_0_out_channels=160,
                                    module0_0_conv2d_2_in_channels=160,
                                    module0_0_conv2d_2_out_channels=160,
                                    module0_0_conv2d_5_in_channels=160,
                                    module0_0_conv2d_5_out_channels=160,
                                    module0_0_conv2d_7_in_channels=160,
                                    module0_0_conv2d_7_out_channels=160,
                                    module0_1_conv2d_0_in_channels=160,
                                    module0_1_conv2d_0_out_channels=160,
                                    module0_1_conv2d_2_in_channels=160,
                                    module0_1_conv2d_2_out_channels=160,
                                    module0_1_conv2d_5_in_channels=160,
                                    module0_1_conv2d_5_out_channels=160,
                                    module0_1_conv2d_7_in_channels=160,
                                    module0_1_conv2d_7_out_channels=160)
        self.module7_7 = Module7(conv2d_0_in_channels=80,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.module7_8 = Module7(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=40,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_324 = nn.ReLU()
        self.conv2d_306 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_9 = Module7(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=80,
                                 resizenearestneighbor_1_size=(28, 28))
        self.relu_322 = nn.ReLU()
        self.module11_2 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_309 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_326 = nn.ReLU()
        self.module16_11 = Module16(module0_0_conv2d_0_in_channels=40,
                                    module0_0_conv2d_0_out_channels=40,
                                    module0_0_conv2d_2_in_channels=40,
                                    module0_0_conv2d_2_out_channels=40,
                                    module0_0_conv2d_5_in_channels=40,
                                    module0_0_conv2d_5_out_channels=40,
                                    module0_0_conv2d_7_in_channels=40,
                                    module0_0_conv2d_7_out_channels=40,
                                    module0_1_conv2d_0_in_channels=40,
                                    module0_1_conv2d_0_out_channels=40,
                                    module0_1_conv2d_2_in_channels=40,
                                    module0_1_conv2d_2_out_channels=40,
                                    module0_1_conv2d_5_in_channels=40,
                                    module0_1_conv2d_5_out_channels=40,
                                    module0_1_conv2d_7_in_channels=40,
                                    module0_1_conv2d_7_out_channels=40)
        self.module16_12 = Module16(module0_0_conv2d_0_in_channels=80,
                                    module0_0_conv2d_0_out_channels=80,
                                    module0_0_conv2d_2_in_channels=80,
                                    module0_0_conv2d_2_out_channels=80,
                                    module0_0_conv2d_5_in_channels=80,
                                    module0_0_conv2d_5_out_channels=80,
                                    module0_0_conv2d_7_in_channels=80,
                                    module0_0_conv2d_7_out_channels=80,
                                    module0_1_conv2d_0_in_channels=80,
                                    module0_1_conv2d_0_out_channels=80,
                                    module0_1_conv2d_2_in_channels=80,
                                    module0_1_conv2d_2_out_channels=80,
                                    module0_1_conv2d_5_in_channels=80,
                                    module0_1_conv2d_5_out_channels=80,
                                    module0_1_conv2d_7_in_channels=80,
                                    module0_1_conv2d_7_out_channels=80)
        self.module16_13 = Module16(module0_0_conv2d_0_in_channels=160,
                                    module0_0_conv2d_0_out_channels=160,
                                    module0_0_conv2d_2_in_channels=160,
                                    module0_0_conv2d_2_out_channels=160,
                                    module0_0_conv2d_5_in_channels=160,
                                    module0_0_conv2d_5_out_channels=160,
                                    module0_0_conv2d_7_in_channels=160,
                                    module0_0_conv2d_7_out_channels=160,
                                    module0_1_conv2d_0_in_channels=160,
                                    module0_1_conv2d_0_out_channels=160,
                                    module0_1_conv2d_2_in_channels=160,
                                    module0_1_conv2d_2_out_channels=160,
                                    module0_1_conv2d_5_in_channels=160,
                                    module0_1_conv2d_5_out_channels=160,
                                    module0_1_conv2d_7_in_channels=160,
                                    module0_1_conv2d_7_out_channels=160)
        self.module7_10 = Module7(conv2d_0_in_channels=80,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_11 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.relu_402 = nn.ReLU()
        self.conv2d_388 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_12 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.relu_403 = nn.ReLU()
        self.module11_3 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_386 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_407 = nn.ReLU()
        self.module5_4 = Module5(conv2d_0_in_channels=160,
                                 conv2d_0_out_channels=320,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module16_14 = Module16(module0_0_conv2d_0_in_channels=40,
                                    module0_0_conv2d_0_out_channels=40,
                                    module0_0_conv2d_2_in_channels=40,
                                    module0_0_conv2d_2_out_channels=40,
                                    module0_0_conv2d_5_in_channels=40,
                                    module0_0_conv2d_5_out_channels=40,
                                    module0_0_conv2d_7_in_channels=40,
                                    module0_0_conv2d_7_out_channels=40,
                                    module0_1_conv2d_0_in_channels=40,
                                    module0_1_conv2d_0_out_channels=40,
                                    module0_1_conv2d_2_in_channels=40,
                                    module0_1_conv2d_2_out_channels=40,
                                    module0_1_conv2d_5_in_channels=40,
                                    module0_1_conv2d_5_out_channels=40,
                                    module0_1_conv2d_7_in_channels=40,
                                    module0_1_conv2d_7_out_channels=40)
        self.module16_15 = Module16(module0_0_conv2d_0_in_channels=80,
                                    module0_0_conv2d_0_out_channels=80,
                                    module0_0_conv2d_2_in_channels=80,
                                    module0_0_conv2d_2_out_channels=80,
                                    module0_0_conv2d_5_in_channels=80,
                                    module0_0_conv2d_5_out_channels=80,
                                    module0_0_conv2d_7_in_channels=80,
                                    module0_0_conv2d_7_out_channels=80,
                                    module0_1_conv2d_0_in_channels=80,
                                    module0_1_conv2d_0_out_channels=80,
                                    module0_1_conv2d_2_in_channels=80,
                                    module0_1_conv2d_2_out_channels=80,
                                    module0_1_conv2d_5_in_channels=80,
                                    module0_1_conv2d_5_out_channels=80,
                                    module0_1_conv2d_7_in_channels=80,
                                    module0_1_conv2d_7_out_channels=80)
        self.module16_16 = Module16(module0_0_conv2d_0_in_channels=160,
                                    module0_0_conv2d_0_out_channels=160,
                                    module0_0_conv2d_2_in_channels=160,
                                    module0_0_conv2d_2_out_channels=160,
                                    module0_0_conv2d_5_in_channels=160,
                                    module0_0_conv2d_5_out_channels=160,
                                    module0_0_conv2d_7_in_channels=160,
                                    module0_0_conv2d_7_out_channels=160,
                                    module0_1_conv2d_0_in_channels=160,
                                    module0_1_conv2d_0_out_channels=160,
                                    module0_1_conv2d_2_in_channels=160,
                                    module0_1_conv2d_2_out_channels=160,
                                    module0_1_conv2d_5_in_channels=160,
                                    module0_1_conv2d_5_out_channels=160,
                                    module0_1_conv2d_7_in_channels=160,
                                    module0_1_conv2d_7_out_channels=160)
        self.module16_17 = Module16(module0_0_conv2d_0_in_channels=320,
                                    module0_0_conv2d_0_out_channels=320,
                                    module0_0_conv2d_2_in_channels=320,
                                    module0_0_conv2d_2_out_channels=320,
                                    module0_0_conv2d_5_in_channels=320,
                                    module0_0_conv2d_5_out_channels=320,
                                    module0_0_conv2d_7_in_channels=320,
                                    module0_0_conv2d_7_out_channels=320,
                                    module0_1_conv2d_0_in_channels=320,
                                    module0_1_conv2d_0_out_channels=320,
                                    module0_1_conv2d_2_in_channels=320,
                                    module0_1_conv2d_2_out_channels=320,
                                    module0_1_conv2d_5_in_channels=320,
                                    module0_1_conv2d_5_out_channels=320,
                                    module0_1_conv2d_7_in_channels=320,
                                    module0_1_conv2d_7_out_channels=320)
        self.module7_13 = Module7(conv2d_0_in_channels=80,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_14 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_15 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.relu_525 = nn.ReLU()
        self.conv2d_484 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_16 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.module7_17 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.relu_526 = nn.ReLU()
        self.module11_4 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_488 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_18 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=160,
                                  resizenearestneighbor_1_size=(14, 14))
        self.relu_527 = nn.ReLU()
        self.module15_4 = Module15(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=320,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(2, 2),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_in_channels=40,
                                   module5_1_conv2d_0_out_channels=40,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(2, 2),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.module11_5 = Module11(conv2d_0_in_channels=80,
                                   conv2d_0_out_channels=320,
                                   module5_0_conv2d_0_in_channels=80,
                                   module5_0_conv2d_0_out_channels=80,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_500 = nn.Conv2d(in_channels=160,
                                    out_channels=320,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_532 = nn.ReLU()
        self.module16_18 = Module16(module0_0_conv2d_0_in_channels=40,
                                    module0_0_conv2d_0_out_channels=40,
                                    module0_0_conv2d_2_in_channels=40,
                                    module0_0_conv2d_2_out_channels=40,
                                    module0_0_conv2d_5_in_channels=40,
                                    module0_0_conv2d_5_out_channels=40,
                                    module0_0_conv2d_7_in_channels=40,
                                    module0_0_conv2d_7_out_channels=40,
                                    module0_1_conv2d_0_in_channels=40,
                                    module0_1_conv2d_0_out_channels=40,
                                    module0_1_conv2d_2_in_channels=40,
                                    module0_1_conv2d_2_out_channels=40,
                                    module0_1_conv2d_5_in_channels=40,
                                    module0_1_conv2d_5_out_channels=40,
                                    module0_1_conv2d_7_in_channels=40,
                                    module0_1_conv2d_7_out_channels=40)
        self.module16_19 = Module16(module0_0_conv2d_0_in_channels=80,
                                    module0_0_conv2d_0_out_channels=80,
                                    module0_0_conv2d_2_in_channels=80,
                                    module0_0_conv2d_2_out_channels=80,
                                    module0_0_conv2d_5_in_channels=80,
                                    module0_0_conv2d_5_out_channels=80,
                                    module0_0_conv2d_7_in_channels=80,
                                    module0_0_conv2d_7_out_channels=80,
                                    module0_1_conv2d_0_in_channels=80,
                                    module0_1_conv2d_0_out_channels=80,
                                    module0_1_conv2d_2_in_channels=80,
                                    module0_1_conv2d_2_out_channels=80,
                                    module0_1_conv2d_5_in_channels=80,
                                    module0_1_conv2d_5_out_channels=80,
                                    module0_1_conv2d_7_in_channels=80,
                                    module0_1_conv2d_7_out_channels=80)
        self.module16_20 = Module16(module0_0_conv2d_0_in_channels=160,
                                    module0_0_conv2d_0_out_channels=160,
                                    module0_0_conv2d_2_in_channels=160,
                                    module0_0_conv2d_2_out_channels=160,
                                    module0_0_conv2d_5_in_channels=160,
                                    module0_0_conv2d_5_out_channels=160,
                                    module0_0_conv2d_7_in_channels=160,
                                    module0_0_conv2d_7_out_channels=160,
                                    module0_1_conv2d_0_in_channels=160,
                                    module0_1_conv2d_0_out_channels=160,
                                    module0_1_conv2d_2_in_channels=160,
                                    module0_1_conv2d_2_out_channels=160,
                                    module0_1_conv2d_5_in_channels=160,
                                    module0_1_conv2d_5_out_channels=160,
                                    module0_1_conv2d_7_in_channels=160,
                                    module0_1_conv2d_7_out_channels=160)
        self.module16_21 = Module16(module0_0_conv2d_0_in_channels=320,
                                    module0_0_conv2d_0_out_channels=320,
                                    module0_0_conv2d_2_in_channels=320,
                                    module0_0_conv2d_2_out_channels=320,
                                    module0_0_conv2d_5_in_channels=320,
                                    module0_0_conv2d_5_out_channels=320,
                                    module0_0_conv2d_7_in_channels=320,
                                    module0_0_conv2d_7_out_channels=320,
                                    module0_1_conv2d_0_in_channels=320,
                                    module0_1_conv2d_0_out_channels=320,
                                    module0_1_conv2d_2_in_channels=320,
                                    module0_1_conv2d_2_out_channels=320,
                                    module0_1_conv2d_5_in_channels=320,
                                    module0_1_conv2d_5_out_channels=320,
                                    module0_1_conv2d_7_in_channels=320,
                                    module0_1_conv2d_7_out_channels=320)
        self.module7_19 = Module7(conv2d_0_in_channels=80,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_20 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_21 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.relu_647 = nn.ReLU()
        self.conv2d_609 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_22 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.module7_23 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.relu_644 = nn.ReLU()
        self.module11_6 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_613 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_24 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=160,
                                  resizenearestneighbor_1_size=(14, 14))
        self.relu_649 = nn.ReLU()
        self.module15_5 = Module15(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=320,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(2, 2),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_in_channels=40,
                                   module5_1_conv2d_0_out_channels=40,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(2, 2),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.module11_7 = Module11(conv2d_0_in_channels=80,
                                   conv2d_0_out_channels=320,
                                   module5_0_conv2d_0_in_channels=80,
                                   module5_0_conv2d_0_out_channels=80,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_617 = nn.Conv2d(in_channels=160,
                                    out_channels=320,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_658 = nn.ReLU()
        self.module16_22 = Module16(module0_0_conv2d_0_in_channels=40,
                                    module0_0_conv2d_0_out_channels=40,
                                    module0_0_conv2d_2_in_channels=40,
                                    module0_0_conv2d_2_out_channels=40,
                                    module0_0_conv2d_5_in_channels=40,
                                    module0_0_conv2d_5_out_channels=40,
                                    module0_0_conv2d_7_in_channels=40,
                                    module0_0_conv2d_7_out_channels=40,
                                    module0_1_conv2d_0_in_channels=40,
                                    module0_1_conv2d_0_out_channels=40,
                                    module0_1_conv2d_2_in_channels=40,
                                    module0_1_conv2d_2_out_channels=40,
                                    module0_1_conv2d_5_in_channels=40,
                                    module0_1_conv2d_5_out_channels=40,
                                    module0_1_conv2d_7_in_channels=40,
                                    module0_1_conv2d_7_out_channels=40)
        self.module16_23 = Module16(module0_0_conv2d_0_in_channels=80,
                                    module0_0_conv2d_0_out_channels=80,
                                    module0_0_conv2d_2_in_channels=80,
                                    module0_0_conv2d_2_out_channels=80,
                                    module0_0_conv2d_5_in_channels=80,
                                    module0_0_conv2d_5_out_channels=80,
                                    module0_0_conv2d_7_in_channels=80,
                                    module0_0_conv2d_7_out_channels=80,
                                    module0_1_conv2d_0_in_channels=80,
                                    module0_1_conv2d_0_out_channels=80,
                                    module0_1_conv2d_2_in_channels=80,
                                    module0_1_conv2d_2_out_channels=80,
                                    module0_1_conv2d_5_in_channels=80,
                                    module0_1_conv2d_5_out_channels=80,
                                    module0_1_conv2d_7_in_channels=80,
                                    module0_1_conv2d_7_out_channels=80)
        self.module16_24 = Module16(module0_0_conv2d_0_in_channels=160,
                                    module0_0_conv2d_0_out_channels=160,
                                    module0_0_conv2d_2_in_channels=160,
                                    module0_0_conv2d_2_out_channels=160,
                                    module0_0_conv2d_5_in_channels=160,
                                    module0_0_conv2d_5_out_channels=160,
                                    module0_0_conv2d_7_in_channels=160,
                                    module0_0_conv2d_7_out_channels=160,
                                    module0_1_conv2d_0_in_channels=160,
                                    module0_1_conv2d_0_out_channels=160,
                                    module0_1_conv2d_2_in_channels=160,
                                    module0_1_conv2d_2_out_channels=160,
                                    module0_1_conv2d_5_in_channels=160,
                                    module0_1_conv2d_5_out_channels=160,
                                    module0_1_conv2d_7_in_channels=160,
                                    module0_1_conv2d_7_out_channels=160)
        self.module16_25 = Module16(module0_0_conv2d_0_in_channels=320,
                                    module0_0_conv2d_0_out_channels=320,
                                    module0_0_conv2d_2_in_channels=320,
                                    module0_0_conv2d_2_out_channels=320,
                                    module0_0_conv2d_5_in_channels=320,
                                    module0_0_conv2d_5_out_channels=320,
                                    module0_0_conv2d_7_in_channels=320,
                                    module0_0_conv2d_7_out_channels=320,
                                    module0_1_conv2d_0_in_channels=320,
                                    module0_1_conv2d_0_out_channels=320,
                                    module0_1_conv2d_2_in_channels=320,
                                    module0_1_conv2d_2_out_channels=320,
                                    module0_1_conv2d_5_in_channels=320,
                                    module0_1_conv2d_5_out_channels=320,
                                    module0_1_conv2d_7_in_channels=320,
                                    module0_1_conv2d_7_out_channels=320)
        self.module7_25 = Module7(conv2d_0_in_channels=80,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_26 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.module7_27 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=40,
                                  resizenearestneighbor_1_size=(56, 56))
        self.relu_768 = nn.ReLU()
        self.conv2d_733 = nn.Conv2d(in_channels=40,
                                    out_channels=80,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_28 = Module7(conv2d_0_in_channels=160,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.module7_29 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=80,
                                  resizenearestneighbor_1_size=(28, 28))
        self.relu_769 = nn.ReLU()
        self.module11_8 = Module11(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=160,
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_729 = nn.Conv2d(in_channels=80,
                                    out_channels=160,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module7_30 = Module7(conv2d_0_in_channels=320,
                                  conv2d_0_out_channels=160,
                                  resizenearestneighbor_1_size=(14, 14))
        self.relu_770 = nn.ReLU()
        self.module15_6 = Module15(conv2d_0_in_channels=40,
                                   conv2d_0_out_channels=320,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(2, 2),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=40,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_in_channels=40,
                                   module5_1_conv2d_0_out_channels=40,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(2, 2),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.module11_9 = Module11(conv2d_0_in_channels=80,
                                   conv2d_0_out_channels=320,
                                   module5_0_conv2d_0_in_channels=80,
                                   module5_0_conv2d_0_out_channels=80,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_stride=(2, 2),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad")
        self.conv2d_740 = nn.Conv2d(in_channels=160,
                                    out_channels=320,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_782 = nn.ReLU()
        self.module15_7 = Module15(conv2d_0_in_channels=32,
                                   conv2d_0_out_channels=128,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=40,
                                   module5_0_conv2d_0_out_channels=32,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=32,
                                   module5_1_conv2d_0_out_channels=32,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.conv2d_773 = nn.Conv2d(in_channels=40,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_800 = nn.ReLU()
        self.module15_8 = Module15(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=80,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=64,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.conv2d_775 = nn.Conv2d(in_channels=80,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_801 = nn.ReLU()
        self.module5_5 = Module5(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module15_9 = Module15(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_in_channels=160,
                                   module5_0_conv2d_0_out_channels=128,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_stride=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_1_conv2d_0_in_channels=128,
                                   module5_1_conv2d_0_out_channels=128,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_stride=(1, 1),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad")
        self.conv2d_777 = nn.Conv2d(in_channels=160,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_802 = nn.ReLU()
        self.module5_6 = Module5(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module15_10 = Module15(conv2d_0_in_channels=256,
                                    conv2d_0_out_channels=1024,
                                    conv2d_0_kernel_size=(1, 1),
                                    conv2d_0_stride=(1, 1),
                                    conv2d_0_padding=0,
                                    conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_in_channels=320,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_stride=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=256,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_stride=(1, 1),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad")
        self.conv2d_787 = nn.Conv2d(in_channels=320,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_807 = nn.ReLU()
        self.module5_7 = Module5(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=1024,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module5_8 = Module5(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=2048,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.avgpool2d_817 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_818 = nn.Flatten()
        self.dense_819 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module5_0_opt = self.module5_0(opt_relu_1)
        module15_0_opt = self.module15_0(module5_0_opt)
        opt_conv2d_5 = self.conv2d_5(module5_0_opt)
        opt_add_10 = P.Add()(module15_0_opt, opt_conv2d_5)
        opt_relu_11 = self.relu_11(opt_add_10)
        module15_1_opt = self.module15_1(opt_relu_11)
        opt_add_17 = P.Add()(module15_1_opt, opt_relu_11)
        opt_relu_18 = self.relu_18(opt_add_17)
        module15_2_opt = self.module15_2(opt_relu_18)
        opt_add_24 = P.Add()(module15_2_opt, opt_relu_18)
        opt_relu_25 = self.relu_25(opt_add_24)
        module15_3_opt = self.module15_3(opt_relu_25)
        opt_add_31 = P.Add()(module15_3_opt, opt_relu_25)
        opt_relu_32 = self.relu_32(opt_add_31)
        module5_1_opt = self.module5_1(opt_relu_32)
        module5_2_opt = self.module5_2(opt_relu_32)
        module16_0_opt = self.module16_0(module5_1_opt)
        module16_1_opt = self.module16_1(module5_2_opt)
        module7_0_opt = self.module7_0(module16_1_opt)
        opt_add_82 = P.Add()(module16_0_opt, module7_0_opt)
        opt_relu_85 = self.relu_85(opt_add_82)
        opt_conv2d_77 = self.conv2d_77(module16_0_opt)
        opt_add_79 = P.Add()(opt_conv2d_77, module16_1_opt)
        opt_relu_81 = self.relu_81(opt_add_79)
        module5_3_opt = self.module5_3(opt_relu_81)
        module16_2_opt = self.module16_2(opt_relu_85)
        module16_3_opt = self.module16_3(opt_relu_81)
        module16_4_opt = self.module16_4(module5_3_opt)
        module7_1_opt = self.module7_1(module16_3_opt)
        opt_add_153 = P.Add()(module16_2_opt, module7_1_opt)
        module7_2_opt = self.module7_2(module16_4_opt)
        opt_add_159 = P.Add()(opt_add_153, module7_2_opt)
        opt_relu_162 = self.relu_162(opt_add_159)
        opt_conv2d_149 = self.conv2d_149(module16_2_opt)
        opt_add_154 = P.Add()(opt_conv2d_149, module16_3_opt)
        module7_3_opt = self.module7_3(module16_4_opt)
        opt_add_160 = P.Add()(opt_add_154, module7_3_opt)
        opt_relu_163 = self.relu_163(opt_add_160)
        module11_0_opt = self.module11_0(module16_2_opt)
        opt_conv2d_145 = self.conv2d_145(module16_3_opt)
        opt_add_161 = P.Add()(module11_0_opt, opt_conv2d_145)
        opt_add_164 = P.Add()(opt_add_161, module16_4_opt)
        opt_relu_167 = self.relu_167(opt_add_164)
        module16_5_opt = self.module16_5(opt_relu_162)
        module16_6_opt = self.module16_6(opt_relu_163)
        module16_7_opt = self.module16_7(opt_relu_167)
        module7_4_opt = self.module7_4(module16_6_opt)
        opt_add_236 = P.Add()(module16_5_opt, module7_4_opt)
        module7_5_opt = self.module7_5(module16_7_opt)
        opt_add_240 = P.Add()(opt_add_236, module7_5_opt)
        opt_relu_243 = self.relu_243(opt_add_240)
        opt_conv2d_225 = self.conv2d_225(module16_5_opt)
        opt_add_230 = P.Add()(opt_conv2d_225, module16_6_opt)
        module7_6_opt = self.module7_6(module16_7_opt)
        opt_add_241 = P.Add()(opt_add_230, module7_6_opt)
        opt_relu_244 = self.relu_244(opt_add_241)
        module11_1_opt = self.module11_1(module16_5_opt)
        opt_conv2d_228 = self.conv2d_228(module16_6_opt)
        opt_add_239 = P.Add()(module11_1_opt, opt_conv2d_228)
        opt_add_242 = P.Add()(opt_add_239, module16_7_opt)
        opt_relu_245 = self.relu_245(opt_add_242)
        module16_8_opt = self.module16_8(opt_relu_243)
        module16_9_opt = self.module16_9(opt_relu_244)
        module16_10_opt = self.module16_10(opt_relu_245)
        module7_7_opt = self.module7_7(module16_9_opt)
        opt_add_318 = P.Add()(module16_8_opt, module7_7_opt)
        module7_8_opt = self.module7_8(module16_10_opt)
        opt_add_321 = P.Add()(opt_add_318, module7_8_opt)
        opt_relu_324 = self.relu_324(opt_add_321)
        opt_conv2d_306 = self.conv2d_306(module16_8_opt)
        opt_add_312 = P.Add()(opt_conv2d_306, module16_9_opt)
        module7_9_opt = self.module7_9(module16_10_opt)
        opt_add_319 = P.Add()(opt_add_312, module7_9_opt)
        opt_relu_322 = self.relu_322(opt_add_319)
        module11_2_opt = self.module11_2(module16_8_opt)
        opt_conv2d_309 = self.conv2d_309(module16_9_opt)
        opt_add_320 = P.Add()(module11_2_opt, opt_conv2d_309)
        opt_add_323 = P.Add()(opt_add_320, module16_10_opt)
        opt_relu_326 = self.relu_326(opt_add_323)
        module16_11_opt = self.module16_11(opt_relu_324)
        module16_12_opt = self.module16_12(opt_relu_322)
        module16_13_opt = self.module16_13(opt_relu_326)
        module7_10_opt = self.module7_10(module16_12_opt)
        opt_add_395 = P.Add()(module16_11_opt, module7_10_opt)
        module7_11_opt = self.module7_11(module16_13_opt)
        opt_add_399 = P.Add()(opt_add_395, module7_11_opt)
        opt_relu_402 = self.relu_402(opt_add_399)
        opt_conv2d_388 = self.conv2d_388(module16_11_opt)
        opt_add_393 = P.Add()(opt_conv2d_388, module16_12_opt)
        module7_12_opt = self.module7_12(module16_13_opt)
        opt_add_400 = P.Add()(opt_add_393, module7_12_opt)
        opt_relu_403 = self.relu_403(opt_add_400)
        module11_3_opt = self.module11_3(module16_11_opt)
        opt_conv2d_386 = self.conv2d_386(module16_12_opt)
        opt_add_401 = P.Add()(module11_3_opt, opt_conv2d_386)
        opt_add_404 = P.Add()(opt_add_401, module16_13_opt)
        opt_relu_407 = self.relu_407(opt_add_404)
        module5_4_opt = self.module5_4(opt_relu_407)
        module16_14_opt = self.module16_14(opt_relu_402)
        module16_15_opt = self.module16_15(opt_relu_403)
        module16_16_opt = self.module16_16(opt_relu_407)
        module16_17_opt = self.module16_17(module5_4_opt)
        module7_13_opt = self.module7_13(module16_15_opt)
        opt_add_503 = P.Add()(module16_14_opt, module7_13_opt)
        module7_14_opt = self.module7_14(module16_16_opt)
        opt_add_513 = P.Add()(opt_add_503, module7_14_opt)
        module7_15_opt = self.module7_15(module16_17_opt)
        opt_add_521 = P.Add()(opt_add_513, module7_15_opt)
        opt_relu_525 = self.relu_525(opt_add_521)
        opt_conv2d_484 = self.conv2d_484(module16_14_opt)
        opt_add_492 = P.Add()(opt_conv2d_484, module16_15_opt)
        module7_16_opt = self.module7_16(module16_16_opt)
        opt_add_514 = P.Add()(opt_add_492, module7_16_opt)
        module7_17_opt = self.module7_17(module16_17_opt)
        opt_add_522 = P.Add()(opt_add_514, module7_17_opt)
        opt_relu_526 = self.relu_526(opt_add_522)
        module11_4_opt = self.module11_4(module16_14_opt)
        opt_conv2d_488 = self.conv2d_488(module16_15_opt)
        opt_add_508 = P.Add()(module11_4_opt, opt_conv2d_488)
        opt_add_515 = P.Add()(opt_add_508, module16_16_opt)
        module7_18_opt = self.module7_18(module16_17_opt)
        opt_add_523 = P.Add()(opt_add_515, module7_18_opt)
        opt_relu_527 = self.relu_527(opt_add_523)
        module15_4_opt = self.module15_4(module16_14_opt)
        module11_5_opt = self.module11_5(module16_15_opt)
        opt_add_520 = P.Add()(module15_4_opt, module11_5_opt)
        opt_conv2d_500 = self.conv2d_500(module16_16_opt)
        opt_add_524 = P.Add()(opt_add_520, opt_conv2d_500)
        opt_add_528 = P.Add()(opt_add_524, module16_17_opt)
        opt_relu_532 = self.relu_532(opt_add_528)
        module16_18_opt = self.module16_18(opt_relu_525)
        module16_19_opt = self.module16_19(opt_relu_526)
        module16_20_opt = self.module16_20(opt_relu_527)
        module16_21_opt = self.module16_21(opt_relu_532)
        module7_19_opt = self.module7_19(module16_19_opt)
        opt_add_631 = P.Add()(module16_18_opt, module7_19_opt)
        module7_20_opt = self.module7_20(module16_20_opt)
        opt_add_639 = P.Add()(opt_add_631, module7_20_opt)
        module7_21_opt = self.module7_21(module16_21_opt)
        opt_add_643 = P.Add()(opt_add_639, module7_21_opt)
        opt_relu_647 = self.relu_647(opt_add_643)
        opt_conv2d_609 = self.conv2d_609(module16_18_opt)
        opt_add_619 = P.Add()(opt_conv2d_609, module16_19_opt)
        module7_22_opt = self.module7_22(module16_20_opt)
        opt_add_633 = P.Add()(opt_add_619, module7_22_opt)
        module7_23_opt = self.module7_23(module16_21_opt)
        opt_add_640 = P.Add()(opt_add_633, module7_23_opt)
        opt_relu_644 = self.relu_644(opt_add_640)
        module11_6_opt = self.module11_6(module16_18_opt)
        opt_conv2d_613 = self.conv2d_613(module16_19_opt)
        opt_add_637 = P.Add()(module11_6_opt, opt_conv2d_613)
        opt_add_641 = P.Add()(opt_add_637, module16_20_opt)
        module7_24_opt = self.module7_24(module16_21_opt)
        opt_add_645 = P.Add()(opt_add_641, module7_24_opt)
        opt_relu_649 = self.relu_649(opt_add_645)
        module15_5_opt = self.module15_5(module16_18_opt)
        module11_7_opt = self.module11_7(module16_19_opt)
        opt_add_646 = P.Add()(module15_5_opt, module11_7_opt)
        opt_conv2d_617 = self.conv2d_617(module16_20_opt)
        opt_add_650 = P.Add()(opt_add_646, opt_conv2d_617)
        opt_add_654 = P.Add()(opt_add_650, module16_21_opt)
        opt_relu_658 = self.relu_658(opt_add_654)
        module16_22_opt = self.module16_22(opt_relu_647)
        module16_23_opt = self.module16_23(opt_relu_644)
        module16_24_opt = self.module16_24(opt_relu_649)
        module16_25_opt = self.module16_25(opt_relu_658)
        module7_25_opt = self.module7_25(module16_23_opt)
        opt_add_745 = P.Add()(module16_22_opt, module7_25_opt)
        module7_26_opt = self.module7_26(module16_24_opt)
        opt_add_752 = P.Add()(opt_add_745, module7_26_opt)
        module7_27_opt = self.module7_27(module16_25_opt)
        opt_add_764 = P.Add()(opt_add_752, module7_27_opt)
        opt_relu_768 = self.relu_768(opt_add_764)
        opt_conv2d_733 = self.conv2d_733(module16_22_opt)
        opt_add_742 = P.Add()(opt_conv2d_733, module16_23_opt)
        module7_28_opt = self.module7_28(module16_24_opt)
        opt_add_753 = P.Add()(opt_add_742, module7_28_opt)
        module7_29_opt = self.module7_29(module16_25_opt)
        opt_add_765 = P.Add()(opt_add_753, module7_29_opt)
        opt_relu_769 = self.relu_769(opt_add_765)
        module11_8_opt = self.module11_8(module16_22_opt)
        opt_conv2d_729 = self.conv2d_729(module16_23_opt)
        opt_add_757 = P.Add()(module11_8_opt, opt_conv2d_729)
        opt_add_762 = P.Add()(opt_add_757, module16_24_opt)
        module7_30_opt = self.module7_30(module16_25_opt)
        opt_add_766 = P.Add()(opt_add_762, module7_30_opt)
        opt_relu_770 = self.relu_770(opt_add_766)
        module15_6_opt = self.module15_6(module16_22_opt)
        module11_9_opt = self.module11_9(module16_23_opt)
        opt_add_767 = P.Add()(module15_6_opt, module11_9_opt)
        opt_conv2d_740 = self.conv2d_740(module16_24_opt)
        opt_add_771 = P.Add()(opt_add_767, opt_conv2d_740)
        opt_add_778 = P.Add()(opt_add_771, module16_25_opt)
        opt_relu_782 = self.relu_782(opt_add_778)
        module15_7_opt = self.module15_7(opt_relu_768)
        opt_conv2d_773 = self.conv2d_773(opt_relu_768)
        opt_add_796 = P.Add()(module15_7_opt, opt_conv2d_773)
        opt_relu_800 = self.relu_800(opt_add_796)
        module15_8_opt = self.module15_8(opt_relu_769)
        opt_conv2d_775 = self.conv2d_775(opt_relu_769)
        opt_add_797 = P.Add()(module15_8_opt, opt_conv2d_775)
        opt_relu_801 = self.relu_801(opt_add_797)
        module5_5_opt = self.module5_5(opt_relu_800)
        opt_add_808 = P.Add()(opt_relu_801, module5_5_opt)
        module15_9_opt = self.module15_9(opt_relu_770)
        opt_conv2d_777 = self.conv2d_777(opt_relu_770)
        opt_add_798 = P.Add()(module15_9_opt, opt_conv2d_777)
        opt_relu_802 = self.relu_802(opt_add_798)
        module5_6_opt = self.module5_6(opt_add_808)
        opt_add_811 = P.Add()(opt_relu_802, module5_6_opt)
        module15_10_opt = self.module15_10(opt_relu_782)
        opt_conv2d_787 = self.conv2d_787(opt_relu_782)
        opt_add_805 = P.Add()(module15_10_opt, opt_conv2d_787)
        opt_relu_807 = self.relu_807(opt_add_805)
        module5_7_opt = self.module5_7(opt_add_811)
        opt_add_814 = P.Add()(opt_relu_807, module5_7_opt)
        module5_8_opt = self.module5_8(opt_add_814)
        opt_avgpool2d_817 = self.avgpool2d_817(module5_8_opt)
        opt_flatten_818 = self.flatten_818(opt_avgpool2d_817)
        opt_dense_819 = self.dense_819(opt_flatten_818)
        return opt_dense_819
