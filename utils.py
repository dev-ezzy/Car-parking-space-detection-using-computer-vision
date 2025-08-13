# utils.py
import cv2
import numpy as np
import json
from shapely.geometry import Polygon, box
from skimage.measure import label, regionprops
import os
from collections import deque

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

def get_parking_spots_from_mask(mask_path, min_area=2000):
    """
    Read binary mask (white = parking spots) and return list of polygons and bboxes.
    Returns list of dicts: {slot_id:int, label:0 (unknown), polygon:[(x,y)...], bbox:[x1,y1,x2,y2], centroid:[x,y]}
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    # threshold to binary
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # label connected regions
    lbl = label(bw // 255)
    props = regionprops(lbl)
    slots = []
    sid = 0
    for prop in props:
        if prop.area < min_area:  # filter tiny spots (tune)
            continue
        # bounding box
        minr, minc, maxr, maxc = prop.bbox  # (row_min, col_min, row_max, col_max)
        bbox = [int(minc), int(minr), int(maxc), int(maxr)]  # x1,y1,x2,y2
        # extract polygon contour
        # mask of this region:
        region_mask = (lbl == prop.label).astype('uint8') * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # contour coordinates are relative to the region bounding box; convert to original coords
        cnt = contours[0].squeeze()
        if cnt.ndim != 2 or cnt.shape[0] < 4:
            continue
        polygon = [(int(pt[0] + minc), int(pt[1] + minr)) for pt in cnt]
        centroid = [int(prop.centroid[1]), int(prop.centroid[0])]
        slots.append({
            "slot_id": sid,
            "polygon": polygon,
            "bbox": bbox,
            "centroid": centroid
        })
        sid += 1
    return slots

def bbox_from_polygon(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]

def crop_patch_from_polygon(image, polygon, dst_size=(224,224)):
    # polygon expected as list of (x,y)
    # we will approximate with 4-point bounding quad via cv2.minAreaRect or use bounding box if polygon not 4
    poly = np.array(polygon, dtype=np.float32)
    if poly.shape[0] >= 4:
        # choose 4 extreme points via convex hull + approx poly
        rect = cv2.minAreaRect(poly)
        box_pts = cv2.boxPoints(rect)  # float32 4x2
        src = np.array(box_pts, dtype=np.float32)
    else:
        x1,y1,x2,y2 = bbox_from_polygon(polygon)
        src = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    dst_w, dst_h = dst_size
    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(image, M, dst_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return patch

class TemporalSmoother:
    def __init__(self, k=5):
        self.k = k
        self.buffers = {}  # slot_id -> deque

    def update(self, slot_id, pred):
        if slot_id not in self.buffers:
            self.buffers[slot_id] = deque(maxlen=self.k)
        self.buffers[slot_id].append(int(pred))
        return int(sum(self.buffers[slot_id]) > (len(self.buffers[slot_id]) / 2))

# Optional: mapping detection bbox to slot via IoU or overlap
def iou_box(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if areaA + areaB - interArea == 0:
        return 0.0
    return interArea / float(areaA + areaB - interArea)

def map_detections_to_slots(detections, slots, iou_thresh=0.2):
    """
    detections: list of boxes [x1,y1,x2,y2]
    slots: list of slot dicts with 'bbox'
    returns: dict slot_id->occupied (True/False)
    """
    occ = {s['slot_id']: False for s in slots}
    for d in detections:
        for s in slots:
            if iou_box(d, s['bbox']) > iou_thresh:
                occ[s['slot_id']] = True
    return occ
