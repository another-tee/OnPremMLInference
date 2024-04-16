class ObjectDetector:

    def input_spec(self):
        print("ObjectDetector :: input_spec")
        pass

    def output_spec(self):
        print("ObjectDetector :: output_spec")
        pass

    def infer(self, batch):
        print("ObjectDetector :: infer")
    
    def process(self, batch, scales, nms_threshold):
        print("ObjectDetector :: process")
        pass