from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers.utils import logging
import os

logger = logging.get_logger(__name__)


def _create_quantized_graph(quantizer, model, graph_path, feature):
    logger.info(f"Creating quantized graph from {graph_path.as_posix()}")
    quantizer.fit(model.config.name_or_path, output_dir=str(graph_path.parent.as_posix()),
                  feature=feature)


def _warmup_onnx_graph(self, n=10):
    for _ in range(n):
        self.__call__(*self.example)


def _forward_onnx(onnx_model, inputs, return_tensors=False):
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
    predictions = onnx_model.run(None, inputs_onnx)
    return predictions


def _export_onnx_graph(quantizer, model, graph_path, feature):
    # if graph exists, but we are here then it means something went wrong in previous load
    # so delete old graph
    if graph_path.exists():
        graph_path.unlink()

    # create parent dir
    if not graph_path.parent.exists():
        os.makedirs(graph_path.parent.as_posix())

    logger.info(f"Saving onnx graph at {graph_path.as_posix()}")

    quantizer.export(model.config.name_or_path, output_path=graph_path,
                     feature=feature)


def create_model_for_providers(model_path: str) -> InferenceSession:
    logger.info(f"Creating model for providers: {model_path}")
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(str(model_path), options)
    session.disable_fallback()

    return session
