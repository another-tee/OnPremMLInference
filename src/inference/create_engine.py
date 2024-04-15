# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import sys
import argparse
from argparse import Namespace

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
def main(args) -> None:
    model_dir = os.path.join("models", args.type, args.model)
    onnx_path = os.path.join(model_dir, "saved_model.onnx")
    trt_engine_path = os.path.join(model_dir, "engine.trt")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError("The ONNX file does not exist.")

    if args.source == "tfod":
        sys.path.append("engine/tfod")
        from engine.tfod import build_engine
    elif args.source == "efficientdet":
        sys.path.append("engine/efficientdet")
        from engine.efficientdet import build_engine
    elif args.source == "keras":
        sys.path.append("engine/onnxsave")
        from engine.onnxsave import build_engine
    else:
        raise ImportError("The model source is not supported.")
    
    # Build engine
    inside_args = Namespace(
        onnx=onnx_path, 
        engine=trt_engine_path,
        batch_size=1,   # effect only keras/efficientdet
        dynamic_batch_size=None,    # effect only efficientdet
        precision="fp16",
        verbose=False,
        workspace=8,
        calib_input=None,
        calib_cache=None,
        calib_num_images=5000,
        calib_batch_size=8
    )
    build_engine.main(inside_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', 
        '--type',
        type=str, 
        default=None, 
        required=True,
        help="Insert the Model type: classification | detection"
    )
    parser.add_argument(
        '-m', 
        '--model',
        type=str, 
        default=None, 
        required=True,
        help="Insert the model name"
    )
    parser.add_argument(
        '-s', 
        '--source',
        type=str,
        default="tfod", 
        required=True,
        choices=["tfod", "efficientdet", "keras"],
        help="Insert the model source name, either tfod/efficientdet/keras, default: tfod"
    )
    args = parser.parse_args()
    main(args)