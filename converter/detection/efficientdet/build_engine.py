# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import sys
import logging
import numpy as np
import tensorrt as trt
from cuda import cudart
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(1, os.path.join(root_dir, os.pardir))

from utils import common
from utils.image_batcher import ImageBatcher
from utils.engine_calibrator import EngineCalibrator

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

# --------------------------------------------------------------------------- #
#                         Define functions/classes                            #
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

        self.network = None
        self.parser = None

    def create_network(self, onnx_path, batch_size, dynamic_batch_size=None):
        """
        Parse the ONNX graph and create the corresponding 
        TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        :param batch_size: Static batch size to build the engine with.
        :param dynamic_batch_size: Dynamic batch size to build the engine with, 
        if given, batch_size is ignored, pass as a comma-separated string 
        or int list as MIN,OPT,MAX
        """
        network_flags = (
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        log.info("Network Description")

        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        profile = self.builder.create_optimization_profile()
        dynamic_inputs = False
        for input in inputs:
            log.info(
                "Input '{}' with shape {} and dtype {}".\
                    format(input.name, input.shape, input.dtype)
            )
            if input.shape[0] == -1:
                dynamic_inputs = True
                if dynamic_batch_size:
                    if type(dynamic_batch_size) is str:
                        dynamic_batch_size = [
                            int(v) for v in dynamic_batch_size.split(",")
                        ]
                    assert len(dynamic_batch_size) == 3
                    min_shape = [dynamic_batch_size[0]] + list(input.shape[1:])
                    opt_shape = [dynamic_batch_size[1]] + list(input.shape[1:])
                    max_shape = [dynamic_batch_size[2]] + list(input.shape[1:])
                    profile.set_shape(
                        input.name, 
                        min_shape, 
                        opt_shape, 
                        max_shape
                    )
                    log.info(
                        "Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(
                        input.name, min_shape, opt_shape, max_shape)
                    )
                else:
                    shape = [batch_size] + list(input.shape[1:])
                    profile.set_shape(input.name, shape, shape, shape)
                    log.info(
                        "Input '{}' Optimization Profile with shape {}".\
                            format(input.name, shape)
                    )
        if dynamic_inputs:
            self.config.add_optimization_profile(profile)

        outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".\
                    format(output.name, output.shape, output.dtype)
            )

    def set_mixed_precision(self):
        """
        Experimental precision mode.
        Enable mixed-precision mode. When set, the layers defined here 
        will be forced to FP16 to maximize
        INT8 inference accuracy, while having minimal impact on latency.
        """
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # All convolution operations in the first four blocks of the graph 
        # are pinned to FP16.
        # These layers have been manually chosen as they give a good 
        # middle-point between int8 and fp16
        # accuracy in COCO, while maintining almost the same latency 
        # as a normal int8 engine.
        # To experiment with other datasets, or a different balance 
        # between accuracy/latency, you may add or remove blocks.
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            if layer.type == trt.LayerType.CONVOLUTION and any([
                    # AutoML Layer Names:
                    "/stem/" in layer.name,
                    "/blocks_0/" in layer.name,
                    "/blocks_1/" in layer.name,
                    "/blocks_2/" in layer.name,
                    # TFOD Layer Names:
                    "/stem_conv2d/" in layer.name,
                    "/stack_0/block_0/" in layer.name,
                    "/stack_1/block_0/" in layer.name,
                    "/stack_1/block_1/" in layer.name,
                ]):
                self.network.get_layer(i).precision = trt.DataType.HALF
                log.info(
                    "Mixed-Precision Layer {} set to HALF STRICT data type".\
                        format(layer.name)
                )

    def create_engine(
            self, 
            engine_path, 
            precision, 
            calib_input=None, 
            calib_cache=None, 
            calib_num_images=5000,
            calib_batch_size=8):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, 
            either 'fp32', 'fp16', 'int8', or 'mixed'.
        :param calib_input: The path to a directory holding 
            the calibration images.
        :param calib_cache: The path where to write the calibration cache to, 
            or if it already exists, load it from.
        :param calib_num_images: The maximum number of images 
            to use for calibration.
        :param calib_batch_size: The batch size to use for 
            the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]

        if precision in ["fp16", "int8", "mixed"]:
            if not self.builder.platform_has_fast_fp16:
                log.warning(
                    "FP16 is not supported natively on this platform/device"
                )
            self.config.set_flag(trt.BuilderFlag.FP16)
        if precision in ["int8", "mixed"]:
            if not self.builder.platform_has_fast_int8:
                log.warning(
                    "INT8 is not supported natively on this platform/device"
                )
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.int8_calibrator = EngineCalibrator(calib_cache)
            if calib_cache is None or not os.path.exists(calib_cache):
                calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                calib_dtype = trt.nptype(inputs[0].dtype)
                self.config.int8_calibrator.set_image_batcher(
                    ImageBatcher(
                        calib_input, 
                        calib_shape, 
                        calib_dtype, 
                        max_num_images=calib_num_images,
                        exact_batches=True, 
                        shuffle_files=True
                    )
                )

        engine_bytes = None
        try:
            engine_bytes = self.builder.build_serialized_network(
                self.network, self.config
            )
        except AttributeError:
            engine = self.builder.build_engine(self.network, self.config)
            engine_bytes = engine.serialize()
            del engine
        assert engine_bytes
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(
        onnx: str,
        engine: str,
        batch_size=1,
        dynamic_batch_size=None,
        precision="fp16",
        verbose=False,
        workspace=8,
        calib_input=None,
        calib_cache="./calibration.cache",
        calib_num_images=5000,
        calib_batch_size=8):
    """Main file to convert onnx to trt.

    Args:
        onnx (str): The input ONNX model file to load
        engine (str): The output path for the TRT engine
        batch_size (int, optional): The static batch size to build 
            the engine with. Defaults to 1.
        dynamic_batch_size (optional): Enable dynamic batch size by providing 
            a comma-separated MIN,OPT,MAX batch size, if this option is set, 
            --batch_size is ignored, example: -d 1,16,32, default: None, 
            build static engine.
        precision (str, optional): The precision mode to build in, 
            either fp32/fp16/int8/mixed. Defaults to "fp16".
        verbose (bool, optional): Enable more verbose log output. 
            Defaults to False.
        workspace (int, optional): The max memory workspace size to allow in Gb. 
            Defaults to 8.
        calib_input (optional): The directory holding images 
            to use for calibration. Defaults to None.
        calib_cache (str, optional): The file path for INT8 calibration cache 
            to use. Defaults to "./calibration.cache".
        calib_num_images (int, optional): The maximum number of images 
            to use for calibration. Defaults to 5000.
        calib_batch_size (int, optional): The batch size for the calibration 
            process. Defaults to 8.
    """
    if precision in ["int8", "mixed"] and not \
        (calib_input or os.path.exists(calib_cache)):
        log.error(
            "When building in int8 precision, --calib_input or an existing" + \
            "--calib_cache file is required"
        )
        sys.exit(1)
    
    print(f"Build at precision: {precision}")
    print(f"EfficientDet batch size: {batch_size}")
    print(f"Allow workspace (in GiB): {workspace}")
    builder = EngineBuilder(verbose, workspace)
    builder.create_network(onnx, batch_size, dynamic_batch_size)
    if precision == "mixed":
        builder.set_mixed_precision()
    builder.create_engine(
        engine, 
        precision, 
        calib_input, 
        calib_cache, 
        calib_num_images,
        calib_batch_size
    )


if __name__ == "__main__":
    pass