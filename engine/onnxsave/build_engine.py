# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
)
import common

from image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set 
            on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path, batch_size):
        """
        Parse the ONNX graph and create the corresponding 
            TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        :param batch_size: Static batch size to build the engine with.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)
        
        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]

        log.info("Network Description")

        # Set shape
        self.batch_size = batch_size
        profile = self.builder.create_optimization_profile()
        input_layer_name = None
        for input in inputs:
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
            if not input_layer_name:
                input_layer_name = str(input.name)
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        profile.set_shape(
            input_layer_name, 
            (self.batch_size, 224, 224, 3), 
            (self.batch_size, 224, 224, 3), 
            (self.batch_size, 224, 224, 3)
        )
        self.config.add_optimization_profile(profile)
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

    def create_engine(self, engine_path, precision):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32' or 'fp16'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        
        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def main(args):
    print(f"Build at precision: {args.precision}")
    print(f"Allow workspace (in GiB): {args.workspace}")
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx, args.batch_size)
    builder.create_engine(args.engine, args.precision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
        help="The static batch size to build the engine with, default: 1")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16"],
        help="The precision mode to build in, either 'fp32' or 'fp16', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int,
        help="The max memory workspace size to allow in Gb, default: 1")
    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    main(args)