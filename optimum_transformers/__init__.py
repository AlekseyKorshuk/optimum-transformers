# Pipelines
from .pipelines import (
    OptimumQuestionAnsweringPipeline,
    OptimumTextClassificationPipeline,
    OptimumTokenClassificationPipeline,
    OptimumZeroShotClassificationPipeline,
    OptimumFillMaskPipeline,
    OptimumTextGenerationPipeline,
    OptimumFeatureExtractionPipeline,
    pipeline,
)

# Utils
from .utils import (
    is_onnxruntime_available,
    require_onnxruntime
)

# Benchmark
from .benchmark import (
    Benchmark,
)
