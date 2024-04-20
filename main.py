# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import time
import argparse
from PIL import Image
from inference.utils.nms import non_max_suppression
from inference.utils.batcher_detector import DetectorImageBatcher
from inference.utils.batcher_classifier import ClassifierImageBatcher

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
class MLProcessing:
    
    def __init__(
            self,
            detector,
            detector_architecture,
            classifier=None,
            classifier_architecture=None,
            tracker=None) -> None:
        
        self.classifier = None
        self.tracker = None

        print(f"INIT DETECTOR.")
        detector_engine = os.path.join(
            "trained_models/detection", detector, "engine.trt"
        )
        if detector_architecture == "ssdmobilenet":
            from inference.detector.trt_ssdmobilenet import SSDMobileNetDetector
            self.detector = SSDMobileNetDetector(detector_engine)
        elif detector_architecture == "efficientdet":
            from inference.detector.trt_efficientdet import EfficientDetDetector
            self.detector = EfficientDetDetector(detector_engine)
        self.detector.init_model()
        self.detector.bind_engine()
        self.detector_batcher = DetectorImageBatcher(
            *self.detector.input_spec(), str(self.detector))
        
        if classifier:
            print(f"INIT CLASSIFIER.")
            classifier_engine = os.path.join(
                "trained_models/classification", classifier, "engine.trt"
            )
            from inference.classifier.trt_keras import MobileNetClassifier
            self.classifier = MobileNetClassifier(classifier_engine)
            self.classifier.init_model()
            self.classifier.bind_engine()
            self.classifier_batcher = ClassifierImageBatcher(
                *self.classifier.input_spec(), str(classifier_architecture)
            )

        if tracker:
            print(f"INIT TRACKER.")
            ...
    
    def set_params(
            self, 
            detector_confidence=0.5,
            detector_iou=None,
            classifier_confidence=None):
        
        self.detector_confidence = detector_confidence
        self.detector_iou = detector_iou
        self.classifier_confidence = classifier_confidence

    def classify(self, xmin, ymin, xmax, ymax, imagePIL):
        bbox = imagePIL.crop((xmin, ymin, xmax, ymax))
        # Future update: Edit batch here, use for loop because of 
        # yield generator
        for c_batch, c_images, c_scales in \
            self.classifier_batcher.get_batch([bbox]):
            classifications = self.classifier.perform_classification(
                batch=c_batch, 
                scales=c_scales, 
                min_confidence=float(self.classifier_confidence) \
                    if self.classifier_confidence else 0.0
            )

            # each generator has 1 image
            return classifications

    def infer(self, batchPILimage):
        batches_results = []
        # d_batch refers to detection batch, etc.
        for d_batch, d_images, d_scales in \
            self.detector_batcher.get_batch(batchPILimage):
            detections = self.detector.perform_detection(
                batch=d_batch, 
                scales=d_scales, 
                nms_threshold=float(self.detector_confidence)
            )
            a_batch_result = []
            for i in range(len(d_images)):
                a_result = []
                for d in detections[i]:
                    xmin, ymin = d['xmin'], d['ymin']
                    xmax, ymax = d['xmax'], d['ymax']
                    score, classname = d['score'], d['class']
                    
                    # Add classifier
                    if self.classifier:
                        classify_classname = self.classify(
                            xmin, ymin, xmax, ymax, d_images[i]
                        )
                        classname = classify_classname[0].get("class", "human")
        
                    a_result.append([xmin, ymin, xmax, ymax, score, classname])
                
                # DO NMS HERE
                if self.detector_iou:
                    picked = non_max_suppression(a_result, self.detector_iou)
                    a_result = [a_result[i] for i in picked]
                
                a_batch_result.append(a_result)
            batches_results.append(a_batch_result)
        return batches_results


def main(args) -> None:
    
    from PIL import Image, ImageDraw
    def draw_boundingboxes(image_name, image_pil, boundingboxes):
        draw = ImageDraw.Draw(image_pil)
        for box_batch in boundingboxes:
            for box in box_batch[0]:
                xmin, ymin, xmax, ymax, score, classname = box
                draw.rectangle(
                    [(xmin, ymin), (xmax, ymax)], outline='lime', width=5
                )
                text = f"{classname} ({score:.2f})"
                draw.text((xmin, ymin), text, fill='crimson')

        output_dir = os.path.join(args.import_images, "results")
        os.makedirs(output_dir, exist_ok=True)
        image_pil.save(os.path.join(output_dir, image_name))
    
    print(f"RUNNING: {args.product}")
    processor = MLProcessing(
        args.detector, args.detector_arch, args.classifier, 
        args.classifier_arch, args.tracker
    )
    processor.set_params(
        args.detector_confidence, 
        args.detector_iou, 
        args.classifier_confidence
    )
    
    if args.import_images:
        image_num = 0
        image_paths = [
            os.path.join(args.import_images, i) 
            for i in os.listdir(args.import_images) if i.endswith(".jpg")
        ]
        start_time = time.time()
        for image_path in image_paths:
            pil_image = Image.open(image_path)
            # !!!!! image input in batch: based on model engine !!!!!
            results = processor.infer([pil_image])
            print(results)
            
            if args.export_images:
                draw_boundingboxes(
                    os.path.basename(image_path),
                    pil_image,
                    results
                )
            image_num += 1

        end_time = time.time()
        fps = image_num / (end_time - start_time)
        print(f"FPS: {fps}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--product', type=str, default=None,
        help='Specify the product that you want to use.')
    parser.add_argument('-d', '--detector', type=str, default=None,
        help='Specify the model for detection.')
    parser.add_argument('-da', '--detector_arch', type=str, default=None,
        help='Specify the model architecture of detection model.')
    parser.add_argument('-dc', '--detector_confidence', type=float, 
        default=None, help='Specify the confidence of detection model.')
    parser.add_argument('-di', '--detector_iou', type=float, default=None,
        help='Specify the NMS iou threshold of detection model' + \
            'if --detector_nms was specified.')
    parser.add_argument('-c', '--classifier', type=str, default=None,
        help='Specify the model for classification.')
    parser.add_argument('-ca', '--classifier_arch', type=str, default=None,
        help='Specify the model architecture of classification model.')
    parser.add_argument('-cc', '--classifier_confidence', type=float, 
        default=None, help='Specify the confidence of classification model.')
    parser.add_argument('-t', '--tracker', type=str, default=None,
        help='Specify the tracking algorithm for object tracking.')
    parser.add_argument('-i', '--import_images', type=str, default=None,
        help='Specify if you want to import images to run testing.')
    parser.add_argument('-e', '--export_images', action="store_true",
        help='Specify if you want to export images with bounding boxes.')
    args = parser.parse_args()
    main(args)

# python3 main.py -p morning_talk -d human_focus -da efficientdet -dc 0.3 -di 0.8 -c staff_focus -ca "mobilenet" -i test_images