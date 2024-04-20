# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import numpy as np

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
def non_max_suppression(results, iou_threshold):
    """ Suppress overlapping detections
    
    Examples
    --------
        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, iou_threshold, scores)
        >>> detections = [detections[i] for i in indices]
    Parameters
    ----------
    results : list
        list of detection results, 
        [[xmin, ymin, xmax, ymax, score, class], [...]]
    iou_threshold : float
        ROIs that overlap more than this values are suppressed.

    Returns
    -------
    List[v]
        Returns indices of detections that have survived 
        non-maxima suppression.
    """
    
    if len(results) == 0:
        return []

    # Convert to np.array
    result_arr = np.array(results)
    boxes = np.array(result_arr[:, :4])
    scores = np.array(result_arr[:, 4])

    try:
        boxes = boxes.astype(np.float)
        scores = scores.astype(np.float)
    except AttributeError:
        boxes = boxes.astype(float)
        scores = scores.astype(float)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    idxs = idxs[np.where(scores[idxs]>0.1)]

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, 
            np.concatenate(([last], np.where(overlap > iou_threshold)[0]))
        )

    return pick


if __name__ == '__main__':
    pass