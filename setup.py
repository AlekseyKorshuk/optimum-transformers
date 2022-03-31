from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

extras = {"testing": ["pytest", "pytest-xdist", "timeout-decorator", "psutil"],
          "quality": ["black >= 20.8b1", "isort >= 5", "flake8"]}
extras["dev"] = extras["testing"] + extras["quality"]

setup(
    name="optimum_transformers",
    version="0.2.0",
    description="Accelerated nlp pipelines using Transformers, Optimum and ONNX Runtime",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aleksey Korshuk",
    author_email="ale-kor02@mail.ru",
    packages=find_packages(),
    keywords=["ONNX", "onnxruntime", "NLP", "transformer", "transformers", "inference", "fast inference", "Optimum",
              "infinity"],
    license="Apache",
    url="https://github.com/AlekseyKorshuk/optimum-transformers",
    install_requires=['coloredlogs>=15.0,<16.0', 'datasets>=2.0,<3.0', 'matplotlib>=3.5,<4.0', 'onnx>=1.10,<2.0',
                      'onnxruntime>=1.10,<2.0', 'optimum>=1.0,<2.0', 'psutil>=5.9,<6.0', 'seaborn>=0.11,<1.0',
                      'transformers>=4.15,<5.0', 'tokenizers>=0.11.6,<0.12', 'sentencepiece>=0.1.96,<0.2'],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        'Documentation': "https://github.com/AlekseyKorshuk/optimum-transformers",
        'Source': "https://github.com/AlekseyKorshuk/optimum-transformers",
    },
)
