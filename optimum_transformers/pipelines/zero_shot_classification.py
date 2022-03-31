import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
from .base import _forward_onnx, _warmup_onnx_graph
from transformers import ZeroShotClassificationPipeline
import torch


class OptimumZeroShotClassificationPipeline(ZeroShotClassificationPipeline):
    def __init__(self, *args, onnx_model, example, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model
        self.example = example
        _warmup_onnx_graph(self)

    def _forward(self, inputs):
        candidate_label = inputs["candidate_label"]
        sequence = inputs["sequence"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        outputs = SequenceClassifierOutput(logits=torch.tensor(_forward_onnx(self.onnx_model, model_inputs)[0]))

        model_outputs = {
            "candidate_label": candidate_label,
            "sequence": sequence,
            "is_last": inputs["is_last"],
            **outputs,
        }
        return model_outputs
