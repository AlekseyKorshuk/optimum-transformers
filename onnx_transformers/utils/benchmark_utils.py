from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()
    buffer.append(end - start)


@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import numpy as np
import os


def plot_benchmark(results: dict, task: str, model_name: str):
    # Compute average inference time + std
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
    time_results_std = np.std([v.model_inference_time for v in results.values()]) * 1000
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylabel("Avg Inference time (ms)")
    ax.set_title(f"Average inference time (ms) for {task}: {model_name}")
    hbar = ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
    # ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
    try:
        ax.bar_label(hbar, labels=['%.2f ms' % e for e in time_results.values()], label_type='center', fmt='%.2f',
                     color='w', fontsize=18)
    except:
        pass
    plt.show()


def run_benchmark(pipelines: list, num_tests: int, model_input: dict):
    results = {}
    for label, pipeline_ in pipelines:
        # Compute
        time_buffer = []

        result = None
        for _ in trange(num_tests, desc=f"Tracking inference time for {label}"):
            with track_infer_time(time_buffer):
                result = pipeline_(*model_input)

        print(f'Check correctness: {result}')
        # Store the result
        results[label] = OnnxInferenceResult(
            time_buffer,
            None
        )
    return results
