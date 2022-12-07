# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint, load_param_into_net
from ldm.models.clip_zh.simple_tokenizer import WordpieceTokenizer
from ldm.modules.encoders.text_encoder import TextEncoder


class FrozenCLIPEmbedder_ZH(nn.Cell):
    def __init__(self, max_length=77, use_fp16=False):
        super(FrozenCLIPEmbedder_ZH, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.max_length = max_length
        self.tokenizer = WordpieceTokenizer()
        self.transformer = TextEncoder(context_length=77, vocab_size=49408, output_dim=768,
                                       width=768, layers=12, heads=12, dtype=self.dtype)
        self.transformer.set_train(False)

    def tokenize(self, texts):
        SOT_TEXT = "[CLS]"
        EOT_TEXT = "[SEP]"
        CONTEXT_LEN = 77

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = ops.Zeros()((len(all_tokens), CONTEXT_LEN), ms.int64)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > CONTEXT_LEN:
                tokens = tokens[:CONTEXT_LEN - 1] + [eot_token]

            result[i, : len(tokens)] = Tensor(tokens)

        return result

    def encode(self, text):
        batch_encoding = self.tokenize(text)
        outputs = self.transformer(batch_encoding)
        return outputs


if __name__ == '__main__':
    import pickle
    data = pickle.load(open('/data1/niuminzhe/stable_diffusion/text_encoder_inout_full.pkl', 'rb'))
    model = FrozenCLIPEmbedder_ZH()
    param_dict = load_checkpoint('/data1/niuminzhe/stable_diffusion/clip_text_encoder_ms.ckpt')
    load_param_into_net(model.transformer, param_dict)
    encoding, out = model.encode(data['inputs'])
    encoding_numpy = encoding.asnumpy()
    out_numpy = out.asnumpy()
    print(encoding_numpy[0, :])
    print(data['encode_input'][0, :])
    print(out_numpy.shape)
    print(out_numpy[0, 0, :20])
    print(data['outputs'][0, 0, :20])
