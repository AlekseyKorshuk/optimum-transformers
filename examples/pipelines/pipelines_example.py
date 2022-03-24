from onnx_transformers import pipeline

pipe = pipeline("text-classification", use_onnx=True, optimize=True)
pipe("This restaurant is awesome")
