# Benchmarking optimized transformer pipeline

## How to run

### With notebook

You can benchmark pipelines easier with [benchmark_pipelines](./notebooks/benchmark_pipelines.ipynb) notebook.

### With own script

```python
from optimum_transformers import Benchmark

task = "sentiment-analysis"
model_name = "philschmid/MiniLM-L6-H384-uncased-sst2"
num_tests = 100

benchmark = Benchmark(task, model_name)
results = benchmark(num_tests, plot=True)
```

## Result

This is as example plot from the benchmark:

![Resulting plot](result.jpg "Resulting plot")
