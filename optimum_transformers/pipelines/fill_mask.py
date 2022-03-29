from transformers.modeling_outputs import MaskedLMOutput
from .base import _forward_onnx, _warmup_onnx_graph
from transformers import FillMaskPipeline
import torch


class OptimumFillMaskPipeline(FillMaskPipeline):
    def __init__(self, *args, onnx_model, example, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model
        self.example = example
        _warmup_onnx_graph(self)

    def _forward(self, model_inputs):
        model_outputs = MaskedLMOutput(logits=torch.tensor(_forward_onnx(self.onnx_model, model_inputs)[0]))
        model_outputs["input_ids"] = model_inputs["input_ids"]
        return model_outputs
