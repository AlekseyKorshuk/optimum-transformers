from transformers.modeling_outputs import QuestionAnsweringModelOutput
from .base import _forward_onnx, _warmup_onnx_graph
from transformers import QuestionAnsweringPipeline
import torch


class OptimumQuestionAnsweringPipeline(QuestionAnsweringPipeline):
    def __init__(self, *args, onnx_model, example, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model
        self.example = example
        _warmup_onnx_graph(self)

    def _forward(self, inputs):
        example = inputs["example"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        model_output = _forward_onnx(self.onnx_model, model_inputs)
        model_output = QuestionAnsweringModelOutput(
            start_logits=torch.tensor(model_output[0]),
            end_logits=torch.tensor(model_output[1])
        )
        start, end = model_output[:2]
        return {"start": start, "end": end, "example": example, **inputs}
