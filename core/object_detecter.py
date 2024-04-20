if model == 'ssd_mobilenet':
    detector = SSDMobilenet()
    detector.init(model_path, label_path,conf = 0.75, nms_threshold = 0.3, filter_big_boxes = False)
    
                