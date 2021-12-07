"""word tonkenizer"""
from transformers import XLMRobertaTokenizer as OriginalXLMRobertaTokenizer
from transformers import AutoTokenizer as OriginalAutoTokenizer


class XLMRobertaTokenizer(OriginalXLMRobertaTokenizer):
    """
        The original XLMRobertaTokenizer is broken, so fix that ourselves.
        (https://github.com/huggingface/transformers/issues/2976)
    """

    def __init__(self, vocab_file, **kwargs):
        """init fun"""
        super().__init__(vocab_file, **kwargs)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    @property
    def vocab_size(self):
        """vocab size"""
        return len(self.sp_model) + self.fairseq_offset


class AutoTokenizer(OriginalAutoTokenizer):
    """
        A wrapper class of transformers.AutoTokenizer.
        This returns our fixed version of XLMRobertaTokenizer in from_pretrained().
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """load token"""
        if "xlm-roberta" in pretrained_model_name_or_path:
            return XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
