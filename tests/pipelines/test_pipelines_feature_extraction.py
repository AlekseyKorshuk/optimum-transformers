import unittest

from transformers import (
    FEATURE_EXTRACTOR_MAPPING,
    MODEL_MAPPING,
    TF_MODEL_MAPPING,
    LxmertConfig,
)
from optimum_transformers import pipeline, OptimumFeatureExtractionPipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch

from .test_pipelines_common import PipelineTestCaseMeta, ANY


# @is_pipeline_test
class FeatureExtractionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_MAPPING
    tf_model_mapping = TF_MODEL_MAPPING

    @require_torch
    def test_small_model_onnx(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            ANY(list)
        )
    @require_torch
    def test_small_model_onnx_quantized(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="pt", optimize=True
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            ANY(list)
        )

    def get_shape(self, input_, shape=None):
        if shape is None:
            shape = []
        if isinstance(input_, list):
            subshapes = [self.get_shape(in_, shape) for in_ in input_]
            if all(s == 0 for s in subshapes):
                shape.append(len(input_))
            else:
                subshape = subshapes[0]
                shape = [len(input_), *subshape]
        elif isinstance(input_, float):
            return 0
        else:
            raise ValueError("We expect lists of floats, nothing else")
        return shape

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        if tokenizer is None:
            self.skipTest("No tokenizer")
            return
        elif type(model.config) in FEATURE_EXTRACTOR_MAPPING or isinstance(model.config, LxmertConfig):
            self.skipTest("This is a bimodal model, we need to find a more consistent way to switch on those models.")
            return
        elif model.config.is_encoder_decoder:
            self.skipTest(
                """encoder_decoder models are trickier for this pipeline.
                Do we want encoder + decoder inputs to get some featues?
                Do we want encoder only features ?
                For now ignore those.
                """
            )

            return
        feature_extractor = OptimumFeatureExtractionPipeline(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )
        return feature_extractor, ["This is a test", "This is another test"]

    def run_pipeline_test(self, feature_extractor, examples):
        outputs = feature_extractor("This is a test")

        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 1)

        # If we send too small input
        # there's a bug within FunnelModel (output with shape [1, 4, 2, 1] doesn't match the broadcast shape [1, 4, 2, 2])
        outputs = feature_extractor(["This is a test", "Another longer test"])
        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 2)

        outputs = feature_extractor("This is a test" * 100, truncation=True)
        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 1)