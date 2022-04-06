from transformers import Text2TextGenerationPipeline
from transformers.file_utils import is_tf_available

from ..generation_utils import GenerationMixin
from .base import _warmup_onnx_graph

if is_tf_available():
    import tensorflow as tf


class OptimumText2TextGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, *args, onnx_model, example, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model
        self.example = example
        _warmup_onnx_graph(self)
    

    def _forward(self, model_inputs, **generate_kwargs):
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            in_b, input_length = tf.shape(model_inputs["input_ids"]).numpy()

        generate_kwargs["min_length"] = generate_kwargs.get("min_length", self.model.config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", self.model.config.max_length)
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])
        generation_matrix = GenerationMixin(self.model, self.onnx_model)
        output_ids = generation_matrix.generate(**model_inputs, **generate_kwargs)
        out_b = output_ids.shape[0]

        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            output_ids = tf.reshape(output_ids, (in_b, out_b // in_b, *output_ids.shape[1:]))
        return {"output_ids": output_ids}     
