from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

requirements = ["transformers",
                "onnxruntime",
                "psutil",
                "coloredlogs",
                "onnx",
                "optimum",
                "datasets"
                ]

extras = {"testing": ["pytest", "pytest-xdist", "timeout-decorator", "psutil"],
          "quality": ["black >= 20.8b1", "isort >= 5", "flake8"]}
extras["dev"] = extras["testing"] + extras["quality"]

setup(
    name="optimum_transformers",
    version="0.1.0",
    description="Accelerated nlp pipelines using Transformers, Optimum and ONNX Runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aleksey Korshuk",
    author_email="ale-kor02@mail.ru",
    packages=find_packages(),
    keywords=["ONNX", "onnxruntime", "NLP", "transformer", "transformers", "inference", "fast inference", "Optimum",
              "infinity"],
    license="Apache",
    url="https://github.com/AlekseyKorshuk/optimum-transformers",
    install_requires=requirements,
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
