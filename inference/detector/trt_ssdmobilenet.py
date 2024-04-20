# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(1, os.path.join(root_dir, os.pardir))

from inference.utils import common
from converter.detection.ssdmobilenet import build_engine

# --------------------------------------------------------------------------- #
#                         Define functions/classes                            #
# --------------------------------------------------------------------------- #
class SSDMobileNetDetector:
    """Implements inference for the SSDMobileNet TensorRT engine."""
    
    def __init__(
            self, 
            engine_path,
            preprocessor="fixed_shape_resizer",
            detection_type="bbox") -> None:
        
        self.engine_path = engine_path
        self.preprocessor = preprocessor
        self.detection_type = detection_type
    
    def __str__(self):
        return self.preprocessor
    
    def init_model(self):
        """Check the existance of engine.trt"""
        model_dir = os.path.dirname(self.engine_path)
        onnx_path = os.path.join(model_dir, "saved_model.onnx")
        label_map_path = os.path.join(model_dir, "label_map.json")
        
        # Check ONNX
        if not os.path.exists(onnx_path) and not \
            os.path.exists(self.engine_path):
            raise FileNotFoundError("Missing onnx file.")
        
        # Check Labelmap
        if not os.path.exists(label_map_path):
            raise FileNotFoundError("Missing label map file.")
        
        # Check Engine
        if not os.path.exists(self.engine_path):
            build_engine.main(
                onnx=onnx_path,
                engine=self.engine_path
            )
        
        with open(label_map_path, "r") as f:
            labels = json.load(f)
            self.detection_labels = [labels[key] for key in sorted(labels)]

    def bind_engine(self):
        """Load TRT Engine usually in __init__"""
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0: #T
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape 
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype) #T
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype, 
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation, #T
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self) -> tuple:
        """Get the specs for the output tensors of the network. 
        Useful to prepare memory allocations.

        Returns:
            tuple: A tuple with two items per element, 
                the shape and (numpy) datatype of each output tensor.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self) -> list:
        """Get the specs for the output tensors of the network. 
            Useful to prepare memory allocations.

        Returns:
            list: A list with two items per element, 
                the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch) -> list:
        """Execute inference on a batch of images.

        Args:
            batch: A numpy array holding the image batch.

        Returns:
            list: A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        common.memcpy_host_to_device(self.inputs[0]['allocation'], batch) #T
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            common.memcpy_device_to_host(
                self.outputs[o]['host_allocation'], 
                self.outputs[o]['allocation']
            ) #T
        return [o['host_allocation'] for o in self.outputs]

    def perform_detection(self, batch, scales=None, nms_threshold=0.01):

        # Run inference
        outputs = self.infer(batch)

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        
        # One additional output for segmentation masks
        if len(outputs) == 5:
            masks = outputs[4]
        
        detections = []
        normalized = (np.max(boxes) < 2.0)
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue

                # Depending on preprocessor, 
                # box scaling will be slightly different.
                if self.preprocessor == "fixed_shape_resizer":
                    scale_x = self.inputs[0]['shape'][1] \
                        if normalized else 1.0
                    scale_y = self.inputs[0]['shape'][2] \
                        if normalized else 1.0
                    if scales and i < len(scales):
                        scale_x /= scales[i][0]
                        scale_y /= scales[i][1]
                    if self.detection_type == 'bbox':
                        mask = None
                    elif self.detection_type == 'segmentation':
                        mask = masks[i][n]
                        mask = mask > self.iou_threshold
                        mask = mask.astype(np.uint8)
                elif self.preprocessor == "keep_aspect_ratio_resizer":
                    # No segmentation models with keep_aspect_ratio_resizer
                    mask = None
                    scale = self.inputs[0]['shape'][2] \
                        if normalized else 1.0
                    if scales and i < len(scales):
                        scale /= scales[i]
                        scale_y = scale
                        scale_x = scale
                
                # Append to detections
                detections[i].append({
                    'ymin': boxes[i][n][0] * scale_y,
                    'xmin': boxes[i][n][1] * scale_x,
                    'ymax': boxes[i][n][2] * scale_y,
                    'xmax': boxes[i][n][3] * scale_x,
                    'score': scores[i][n],
                    'class': self.detection_labels[int(classes[i][n])],
                    'mask': mask
                })
        
        return detections


if __name__ == '__main__':
    pass
