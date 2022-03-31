# Optimum Transformers

[![Tests](https://github.com/AlekseyKorshuk/optimum-transformers/actions/workflows/python-app.yml/badge.svg)](https://github.com/AlekseyKorshuk/optimum-transformers/actions/workflows/python-app.yml)
[![License](https://img.shields.io/github/license/AlekseyKorshuk/optimum-transformers.svg?color=blue)](https://github.com/AlekseyKorshuk/optimum-transformers/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/optimum-transformers)](https://pypi.org/project/optimum-transformers/)

<img src="https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/social_preview.png?raw=True" width="1200">

Accelerated NLP pipelines for fast inference 🚀 on CPU and GPU. Built with 🤗Transformers, Optimum and ONNX runtime.

## Installation:

With PyPI:

```bash
pip install optimum-transformers
```

Or directly from GitHub:

```bash
pip install git+https://github.com/AlekseyKorshuk/optimum-transformers
```

## Usage:

The pipeline API is similar to transformers [pipeline](https://huggingface.co/transformers/main_classes/pipelines.html)
with just a few differences which are explained below.

Just provide the path/url to the model, and it'll download the model if needed from
the [hub](https://huggingface.co/models) and automatically create onnx graph and run inference.

```python
from optimum_transformers import pipeline

# Initialize a pipeline by passing the task name and 
# set onnx to True (default value is also True)
nlp = pipeline("sentiment-analysis", use_onnx=True)
nlp("Transformers and onnx runtime is an awesome combo!")
# [{'label': 'POSITIVE', 'score': 0.999721109867096}]  
```

Or provide a different model using the `model` argument.

```python
from optimum_transformers import pipeline

nlp = pipeline("question-answering", model="deepset/roberta-base-squad2", use_onnx=True)
nlp(question="What is ONNX Runtime ?",
         context="ONNX Runtime is a highly performant single inference engine for multiple platforms and hardware")
# {'answer': 'highly performant single inference engine for multiple platforms and hardware', 'end': 94,
# 'score': 0.751201868057251, 'start': 18}
```

```python
from optimum_transformers import pipeline

nlp = pipeline("ner", model="mys/electra-base-turkish-cased-ner", use_onnx=True, optimize=True,
                    grouped_entities=True)
nlp("adana kebap ülkemizin önemli lezzetlerinden biridir.")
# [{'entity_group': 'B-food', 'score': 0.869149774312973, 'word': 'adana kebap'}]
```

Set `use_onnx` to `False` for standard torch inference. Set `optimize` to `True` for quantize with ONNX. ( set `use_onnx` to
True)

## Supported pipelines

You can create `Pipeline` objects for the following down-stream tasks:

- `feature-extraction`: Generates a tensor representation for the input sequence
- `ner` and `token-classification`: Generates named entity mapping for each word in the input sequence.
- `sentiment-analysis`: Gives the polarity (positive / negative) of the whole input sequence. Can be used for any text
  classification model.
- `question-answering`: Provided some context and a question referring to the context, it will extract the answer to the
  question in the context.
- `text-classification`: Classifies sequences according to a given number of classes from training.
- `zero-shot-classification`: Classifies sequences according to a given number of classes directly in runtime.
- `fill-mask`: The task of masking tokens in a sequence with a masking token, and prompting the model to fill that mask
  with an appropriate token.
- `text-generation`: The task of generating text according to the previous text provided.

Calling the pipeline for the first time loads the model, creates the onnx graph, and caches it for future use. Due to
this, the first load will take some time. Subsequent calls to the same model will load the onnx graph automatically from
the cache.

## Benchmarks

> Note: For some reason, onnx is slow on colab notebook, so you won't notice any speed-up there. Benchmark it on your own hardware.

Check our example of benchmarking: [example](./examples/benchmark).

For detailed benchmarks and other information refer to this blog post and notebook.

- [Accelerate your NLP pipelines using Hugging Face Transformers and ONNX Runtime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
- [Exporting 🤗 transformers model to ONNX](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb)

> Note: These results were collected on my local machine. So if you have high performance machine to benchmark, please contact me.

**Benchmark `sentiment-analysis` pipeline**

![](https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/sentiment_analysis_benchmark.jpg)

**Benchmark `zero-shot-classification` pipeline**

![](https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/zero_shot_classification_benchmark.jpg)

**Benchmark `token-classification` pipeline**

![](https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/token_classification_benchmark.jpg)

![](https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/token_classification_benchmark2.jpg)

**Benchmark `question-answering` pipeline**

![](https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/question_answering_benchmark.jpg)

**Benchmark `fill-mask` pipeline**

![](https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/fill_mask_benchmark.jpg)

## About

*Built by Aleksey Korshuk*

[![Follow](https://img.shields.io/github/followers/AlekseyKorshuk?style=social)](https://github.com/AlekseyKorshuk)

[![Follow](https://img.shields.io/twitter/follow/alekseykorshuk?style=social)](https://twitter.com/intent/follow?screen_name=alekseykorshuk)

[![Follow](https://img.shields.io/badge/dynamic/json?color=blue&label=Telegram%20Channel&query=%24.result&url=https%3A%2F%2Fapi.telegram.org%2Fbot1929545866%3AAAFGhV-KKnegEcLiyYJxsc4zV6C-bdPEBtQ%2FgetChatMemberCount%3Fchat_id%3D-1001253621662&style=social&logo=telegram)](https://t.me/joinchat/_CQ04KjcJ-4yZTky)

🚀 If you want to contribute to this project OR create something cool together — contact
me: [link](https://github.com/AlekseyKorshuk)

Star this repository:

[![GitHub stars](https://img.shields.io/github/stars/AlekseyKorshuk/optimum-transformers?style=social)](https://github.com/AlekseyKorshuk/optimum-transformers)

## Resources

* Inspired by [Huggingface Infinity](https://huggingface.co/infinity)
* First step done by [Suraj Patil](https://github.com/patil-suraj/onnx_transformers)
* [Optimum](https://huggingface.co/docs/optimum/index)
* [ONNX](https://onnx.ai)
