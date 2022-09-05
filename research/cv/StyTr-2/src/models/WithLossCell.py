from mindspore import nn
from src.utils.function import calc_mean_std, normal


class StyTRWithLossCell(nn.Cell):
    def __init__(self, encoder, stytran, args):
        super(StyTRWithLossCell, self).__init__()

        self.stytran = stytran
        self.enc_1 = encoder[:4]  # input -> relu1_1
        self.enc_2 = encoder[4:11]  # relu1_1 -> relu2_1
        self.enc_3 = encoder[11:18]  # relu2_1 -> relu3_1
        self.enc_4 = encoder[18:31]  # relu3_1 -> relu4_1
        self.enc_5 = encoder[31:44]  # relu4_1 -> relu5_1

        # TODO
        for param in self.enc_1.get_parameters():
            param.requires_grad = False
        for param in self.enc_2.get_parameters():
            param.requires_grad = False
        for param in self.enc_3.get_parameters():
            param.requires_grad = False
        for param in self.enc_4.get_parameters():
            param.requires_grad = False
        for param in self.enc_5.get_parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = stytran.transformer
        self.decode = stytran.decode
        self.embedding = stytran.embedding
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight

    def encode_with_intermediate(self, x):
        x1 = self.enc_1(x)
        x2 = self.enc_2(x1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)

        return [x1, x2, x3, x4, x5]

    def calc_content_loss(self, x, target):

        return self.mse_loss(x, target)

    def calc_style_loss(self, x, target):
        input_mean, input_std = calc_mean_std(x)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def construct(self, content_input, style_input):
        sample_c = content_input
        sample_s = style_input
        content_feats = self.encode_with_intermediate(content_input)
        style_feats = self.encode_with_intermediate(style_input)
        Ics, content, style = self.stytran(content_input, style_input)

        pos_s = None
        pos_c = None

        mask = None
        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
                 self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])

        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
        Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))

        # Identity losses lambda 1
        loss_lambda1 = self.calc_content_loss(Icc, sample_c) + \
                       self.calc_content_loss(Iss, sample_s)
        # Identity losses lambda 2
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)

        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0],
                                                                                                       style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(
                Iss_feats[i], style_feats[i])
        loss_c = self.content_weight * loss_c
        loss_s = self.style_weight * loss_s
        loss = loss_c + loss_s + (loss_lambda1 * 70) + (loss_lambda2 * 1)

        return loss
