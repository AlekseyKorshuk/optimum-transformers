# Pipelines

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of
the complex code from the library, offering a simple API dedicated to several tasks.

## How to use

### Quick start

The usage is exactly the same as original pipelines, except minor improves:

```python
from optimum_transformers import pipeline

pipe = pipeline("text-classification", use_onnx=True, optimize=True)
pipe("This restaurant is awesome")
# [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

* use_onnx - converts default model to ONNX graph
* optimize - optimizes converted ONNX graph with [Optimum](https://huggingface.co/docs/optimum/index)

### Optimum config

Read [Optimum](https://huggingface.co/docs/optimum/index) documentation for more details

```python
from optimum_transformers import pipeline
from optimum.onnxruntime import ORTConfig

ort_config = ORTConfig(quantization_approach="dynamic")
pipe = pipeline("text-classification", use_onnx=True, optimize=True, ort_config=ort_config)
pipe("This restaurant is awesome")
# [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```
