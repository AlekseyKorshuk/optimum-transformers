import contextlib
import inspect
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest
from distutils.util import strtobool
from io import StringIO
from pathlib import Path
from typing import Iterator, Union
from unittest import mock
from transformers import logging as transformers_logging
import importlib.util

_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None


def is_onnxruntime_available():
    return _onnxruntime_available


def require_onnxruntime(test_case):
    if not is_onnxruntime_available():
        return unittest.skip("test requires onnxruntime")(test_case)
    else:
        return test_case

