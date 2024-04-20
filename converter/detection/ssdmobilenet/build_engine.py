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
        :param verbose: If enabled, a higher verbosity level 
            will be set on the TensorRT logger.
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

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding 
        TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
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

        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".\
                    format(input.name, input.shape, input.dtype)
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".\
                    format(output.name, output.shape, output.dtype)
            )
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

        # TODO: These overrides are to improve fp16/int8 performance on 
        # FRCNN models. It might be possible to avoid doing this by 
        # using different box encoding
        # type on the two NMS plugins. To be determined.
        for i in range(self.network.num_layers):
            if self.network.get_layer(i).name in [
                    "FirstStageBoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/squeeze",
                    "FirstStageBoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/scale_value:0",
                    "FirstStageBoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/scale",
                    "nms/anchors:0"]:
                self.network.get_layer(i).precision = trt.DataType.FLOAT
                self.network.get_layer(i-1).precision = trt.DataType.FLOAT
            if self.network.get_layer(i).name == "FirstNMS/detection_boxes_conversion":
                self.network.get_layer(i).precision = trt.DataType.FLOAT

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
            either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, 
            or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]

        # TODO: Strict type is only needed If the per-layer 
        # precision overrides are used
        # If a better method is found to deal with that issue, 
        # this flag can be removed.
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning(
                    "FP16 is not supported natively on this platform/device"
                )
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning(
                    "INT8 is not supported natively on this platform/device"
                )
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more 
                    # efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(
                            calib_input, 
                            calib_shape, 
                            calib_dtype, 
                            max_num_images=calib_num_images,
                            exact_batches=True
                        )
                    )

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def main(
        onnx: str,
        engine: str,
        precision="fp16",
        verbose=False,
        workspace=1,
        calib_input=None,
        calib_cache="./calibration.cache",
        calib_num_images=5000,
        calib_batch_size=8):
    """Main file to convert onnx to trt.

    Args:
        onnx (str): The input ONNX model file to load
        engine (str): The output path for the TRT engine
        precision (str, optional): The precision mode to build in, 
            either 'fp32', 'fp16' or 'int8'. Defaults to "fp16".
        verbose (bool, optional): Enable more verbose log output. 
            Defaults to False.
        workspace (int, optional): The max memory workspace size to allow in Gb. 
            Defaults to 1.
        calib_input (optional): The directory holding images 
            to use for calibration. Defaults to None.
        calib_cache (str, optional): The file path for INT8 calibration cache 
            to use. Defaults to "./calibration.cache".
        calib_num_images (int, optional): The maximum number of images 
            to use for calibration. Defaults to 5000.
        calib_batch_size (int, optional): The batch size for the calibration 
            process. Defaults to 8.
    """
    
    if precision == "int8" and not (calib_input or os.path.exists(calib_cache)):
        log.error(
            "When building in int8 precision, --calib_input or an existing" + \
            "--calib_cache file is required"
        )
        sys.exit(1)
    
    print(f"Build at precision: {precision}")
    print(f"Allow workspace (in GiB): {workspace}")
    builder = EngineBuilder(verbose, workspace)
    builder.create_network(onnx)
    builder.create_engine(
        engine, 
        precision, 
        calib_input, 
        calib_cache, 
        calib_num_images,
        calib_batch_size
    )



if __name__ == '__main__':
    pass