from ..pipelines import pipeline, SUPPORTED_TASKS, TASK_ALIASES
from ..utils.benchmark import plot_benchmark, OnnxInferenceResult, track_infer_time
from tqdm import trange
import numpy as np

PIPELINES = [
    (
        "ONNX Quantized",
        {
            "use_onnx": True,
            "optimize": True
        }
    ),
    (
        "ONNX",
        {
            "use_onnx": True,
            "optimize": False
        }
    ),
    (
        "Pytorch",
        {
            "use_onnx": False,
            "optimize": False
        }
    )
]


class Benchmark:

    def __init__(self,
                 task: str,
                 model: str = None
                 ):
        self.task = task
        if model == "":
            model = None
        self.model = model

    def __call__(self, num_tests: int = 100, model_inputs: dict = None, plot: bool = False):
        if not model_inputs:
            task = self.task
            if self.task in TASK_ALIASES:
                task = TASK_ALIASES[self.task]
            model_inputs = SUPPORTED_TASKS[task]["example"]
        results = self.run_benchmark(num_tests, model_inputs)
        time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}

        if plot:
            plot_benchmark(time_results, self.task, self.model)

        return time_results

    def run_benchmark(self, num_tests: int, model_input: dict):
        results = {}
        for label, pipeline_args in PIPELINES:
            pipeline_ = pipeline(self.task, self.model, **pipeline_args)
            # Compute
            time_buffer = []

            result = None
            for _ in trange(num_tests, desc=f"Tracking inference time for {label}"):
                with track_infer_time(time_buffer):
                    result = pipeline_(**model_input)
            print(f'Check correctness: {result}')
            # Store the result
            results[label] = OnnxInferenceResult(
                time_buffer,
                None
            )

        return results
