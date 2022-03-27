from transformers.modeling_outputs import TokenClassifierOutput
from .base import _forward_onnx, _warmup_onnx_graph
from transformers import TokenClassificationPipeline
import torch


class OptimumTokenClassificationPipeline(TokenClassificationPipeline):
    def __init__(self, *args, onnx_model, example, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model
        self.example = example
        _warmup_onnx_graph(self)

    def preprocess(self, sentence, offset_mapping=None):
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        model_inputs["sentence"] = sentence

        return model_inputs

    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        logits = TokenClassifierOutput(logits=torch.tensor(_forward_onnx(self.onnx_model, model_inputs)[0]))[0]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            **model_inputs,
        }


OptimumNerPipeline = OptimumTokenClassificationPipeline
