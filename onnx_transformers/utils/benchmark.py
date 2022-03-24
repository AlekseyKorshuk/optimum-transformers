from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from ..pipelines import pipeline

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
    # time_results_std = np.std([v.model_inference_time for v in results.values()]) * 1000
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylabel("Avg Inference time (ms)")
    ax.set_title(f"Average inference time (ms) for {task}: {model_name}")
    hbar = ax.bar(results.keys(), results.values(), yerr=None)
    # ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
    try:
        ax.bar_label(hbar, labels=['%.2f ms' % e for e in results.values()], label_type='center', fmt='%.2f',
                     color='w', fontsize=18)
    except:
        pass
    plt.show()



