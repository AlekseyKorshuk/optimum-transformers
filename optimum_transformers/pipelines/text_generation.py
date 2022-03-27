from transformers.file_utils import is_tf_available
from ..generation_utils import GenerationMixin
from .base import _warmup_onnx_graph
from transformers import TextGenerationPipeline

if is_tf_available():
    import tensorflow as tf


class OptimumTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, onnx_model, example, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model
        self.example = example
        _warmup_onnx_graph(self)

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")
        generation_matrix = GenerationMixin(self.model, self.onnx_model)
        generated_sequence = generation_matrix.generate(input_ids=input_ids, **generate_kwargs)  # BS x SL
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}
