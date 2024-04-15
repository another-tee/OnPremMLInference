# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import sys
from argparse import Namespace

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
)
from core.utils.trt_detector import TensorRTDetector
from core.utils.trt_classifier import TensorRTClassifier
from utils.infer_utils import ImageBatcher

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
class UsecaseMorningTalk:
    
    def __init__(self) -> None:
        self.detection_model = "models/detection/human_model"
        self.classification_model = "models/classification/ipad_model"
        self.detector_path = os.path.join(self.detection_model, "engine.trt")
        self.classifier_path = os.path.join(
            self.classification_model, "engine.trt"
        )
        self.detector_label = os.path.join(
            self.detection_model, "label_map.txt"
        )
        self.classifier_label = os.path.join(
            self.classification_model, "label_map.txt"
        )
        
    def init_model(self):

        detector_onnx_path = os.path.join(
            self.detection_model, "saved_model.onnx"
        )
        classifier_onnx_path = os.path.join(
            self.classification_model, "saved_model.onnx"
        )

        # Check ONNX
        if not os.path.exists(detector_onnx_path):
            raise FileNotFoundError("Missing detection onnx.")
        if not os.path.exists(classifier_onnx_path):
            raise FileNotFoundError("Missing classification onnx.")

        # Check Labelmap
        if not os.path.exists(self.detector_label):
            with open(self.detector_label, 'w') as dlabelfile:
                dlabelfile.write("human")
        if not os.path.exists(self.classifier_label):
            with open(self.classifier_label, 'w') as clabelfile:
                clabelfile.write("ipad\n")
                clabelfile.write("no_ipad")

        # Check Engine
        if not os.path.exists(self.detector_path):
            import create_engine
            inside_args = Namespace(
                type="detection", 
                model="human_model",
                source="efficientdet"
            )
            create_engine.main(inside_args)
        if not os.path.exists(self.classifier_path):
            import create_engine
            inside_args = Namespace(
                type="classification", 
                model="ipad_model",
                source="keras"
            )
            create_engine.main(inside_args)
        self.detection_labels = []
        with open(self.detector_label) as dlabel:
            for _, label in enumerate(dlabel):
                self.detection_labels.append(label.strip())
        
        self.classification_labels = []
        with open(self.classifier_label) as clabel:
            for _, label in enumerate(clabel):
                self.classification_labels.append(label.strip())
        
        self.detector = TensorRTDetector(self.detector_path, "efficientdet")
        self.batcher = ImageBatcher(*self.detector.input_spec(), "efficientdet")
        
    def infer(self, image):
        
        # Process a single image
        batch_data, batch_images, batch_scales = self.batcher.get_batch([image])
        detections = self.detector.process(batch_data, batch_scales, 0.5)

        results = []
        for d in detections[0]:
            xmin, ymin, xmax, ymax = d['xmin'], d['ymin'], d['xmax'], d['ymax']
            score, classname = d['score'], self.detection_labels[d['class']]
            results.append([xmin, ymin, xmax, ymax, score, classname])

        return results


# Test
test_obj = UsecaseMorningTalk()
test_obj.init_model()

import time
from PIL import Image
start_time = time.time()
for image_path in os.listdir("usecases/test_images"):
    test_pil_image = Image.open(f"usecases/test_images/{image_path}")
    test_res = test_obj.infer(test_pil_image)
    print(test_res)
end_time = time.time()
print(f"FPS: {1500/(end_time-start_time)}")