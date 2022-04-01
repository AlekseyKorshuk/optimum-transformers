import unittest

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
)
from optimum_transformers import pipeline, OptimumTextClassificationPipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


# @is_pipeline_test
class TextClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

    # @slow
    @require_torch
    def test_small_model_onnx(self):
        text_classifier = pipeline(
            task="text-classification", framework="pt"
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

    # @slow
    @require_torch
    def test_small_model_onnx_quantized(self):
        text_classifier = pipeline(
            task="text-classification", framework="pt",
            optimize=True
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        text_classifier = OptimumTextClassificationPipeline(model=model, tokenizer=tokenizer)
        return text_classifier, ["HuggingFace is in", "This is another test"]

    def run_pipeline_test(self, text_classifier, _):
        model = text_classifier.model
        # Small inputs because BartTokenizer tiny has maximum position embeddings = 22
        valid_inputs = "HuggingFace is in"
        outputs = text_classifier(valid_inputs)

        self.assertEqual(nested_simplify(outputs), [{"label": ANY(str), "score": ANY(float)}])
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())

        valid_inputs = ["HuggingFace is in ", "Paris is in France"]
        outputs = text_classifier(valid_inputs)
        self.assertEqual(
            nested_simplify(outputs),
            [{"label": ANY(str), "score": ANY(float)}, {"label": ANY(str), "score": ANY(float)}],
        )
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())
        self.assertTrue(outputs[1]["label"] in model.config.id2label.values())
