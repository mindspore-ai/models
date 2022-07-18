import numpy as np
import mindspore as ms
import mindspore.nn as nn


class VGG(nn.Cell):
    def __init__(self):
        super(VGG, self).__init__()
        self.net_mode = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract_ = [3, 8, 15, 22, 29, 30]
        #self.extract_ = [1, 4, 8, 12, 16, 17]
        self.in_c = 3
        self.model = self.build_model()

        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(ms.Tensor(np.random.normal(0, np.sqrt(2./n), m.weight.data.shape).astype(np.float32)))
                if m.bias is not None:
                    m.bias.set_data(ms.Tensor(np.zeros(m.bias.data.shape).astype(np.float32)))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(ms.Tensor(np.ones(m.gamma.data.shape).astype(np.float32)))
                m.beta.set_data(ms.Tensor(np.zeros(m.beta.data.shape).astype(np.float32)))


    def build_model(self):
        in_c = self.in_c
        layer = []
        net_layers = []
        for k in self.net_mode:
            if k == "M":
                layer = [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), pad_mode='same')]
            else:
                layer = [nn.Conv2d(in_channels=in_c, out_channels=k, kernel_size=(3, 3), pad_mode='same',
                                   has_bias=False), nn.ReLU()]
                in_c = k
            net_layers += layer
        return nn.CellList(net_layers)

    def construct(self, data):
        outputs = []
        i = 0
        for layer in self.model:
            out = layer(data)
            if i in self.extract_:
                outputs.append(out)
            data = out
            i += 1
        return outputs
