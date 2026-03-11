import re
import json
from PIL import Image
import ast

import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

def extract_content(mark,text):
    pattern = f'<{mark}>(.*?)</{mark}>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        content = match.group(1).strip()
        return content
    else:
        return None
    


def zoom_in(image_path: str, bbox: str, output_path: str):
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except (ValueError, SyntaxError):
            try:
                bbox = json.loads(bbox)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse bbox string: {bbox}") from e
    
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must be a list or tuple containing 4 numbers: {bbox}")
    
    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (ValueError, TypeError) as e:
        raise ValueError(f"bbox coordinates must be numbers: {bbox}") from e

    image = Image.open(image_path)
    
    width, height = image.size
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox coordinates: {bbox}")
    
    cropped_image = image.crop((x1, y1, x2, y2))
    
    if output_path:
        try:
            cropped_image.save(output_path)
            print(f"Cropped image saved to: {output_path}")
        except Exception as e:
            raise IOError(f"Failed to save cropped image: {str(e)}") from e
    
    crop_info = {
        'crop_bbox': [x1, y1, x2, y2],
        'original_size': [width, height],
        'cropped_size': [x2-x1, y2-y1]
    }
    
    return crop_info


def restore_bbox(cropped_bbox, crop_info):
    
    if isinstance(cropped_bbox, str):
        cropped_bbox = ast.literal_eval(cropped_bbox)
    
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_info['crop_bbox']
    original_width, original_height = crop_info['original_size']
    
    crop_box_x1, crop_box_y1, crop_box_x2, crop_box_y2 = cropped_bbox
    
    restored_x1 = crop_box_x1 + crop_x1
    restored_y1 = crop_box_y1 + crop_y1
    restored_x2 = crop_box_x2 + crop_x1
    restored_y2 = crop_box_y2 + crop_y1
    
    restored_x1 = max(0, min(restored_x1, original_width))
    restored_y1 = max(0, min(restored_y1, original_height))
    restored_x2 = max(0, min(restored_x2, original_width))
    restored_y2 = max(0, min(restored_y2, original_height))
    
    return [restored_x1, restored_y1, restored_x2, restored_y2]

def get_gt_pairs(data):
    raw_entities = data['entities']
    entities = []
    for entity in raw_entities:
        entities.append({entity['entity_name'].split('#')[0].strip(): convert_bbox_xywh_to_xyxy(entity['bbox'])})

    gt_pairs = []
    for value in data['relations'].values():
        if value is not None:
            for v in value:
                gt_pairs.append(v)
    for pair in gt_pairs:
        pair[0] = int(pair[0])
        pair[1] = int(pair[1])
            
    return entities, gt_pairs

def convert_bbox_xywh_to_xyxy(bbox):
    
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse bbox string: {bbox}") from e
    
    try:
        x, y, w, h = map(float, bbox)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Bbox coordinates must be numbers: {bbox}") from e
    
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    return [x1, y1, x2, y2]


def convert_bbox_xyxy_to_xywh(bbox):
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse bbox string: {bbox}") from e
    
    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Bbox coordinates must be numbers: {bbox}") from e
    
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    
    return [x, y, w, h]

def calculate_giou(box1, box2):
    """Calculate Generalized IoU (GIoU) between two boxes [x1,y1,x2,y2]"""
    # Convert coordinates to float
    try:
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid box coordinates. Box1: {box1}, Box2: {box2}") from e
    
    # Intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - intersection
    
    # GIoU calculation
    if union == 0:
        return 0
    
    # Convex hull (smallest enclosing box)
    cx1 = min(box1[0], box2[0])
    cy1 = min(box1[1], box2[1])
    cx2 = max(box1[2], box2[2])
    cy2 = max(box1[3], box2[3])
    convex_area = (cx2-cx1)*(cy2-cy1)
    
    iou = intersection / union
    giou = iou - ((convex_area - union) / convex_area)
    return giou

def match_detections_to_gt(detections, gt_boxes, giou_threshold=0.5):
    """
    Match detected objects to GT using Hungarian algorithm with GIoU
    
    Args:
        detections: List of {'label': [x1,y1,x2,y2]} dicts
        gt_boxes: List of {'label': [x1,y1,x2,y2]} dicts
        giou_threshold: Minimum GIoU for valid matches
        
    Returns:
        List of tuples (detection, matched_gt, giou_score) for valid matches
        Unmatched detections are filtered out
    """
    # Convert to lists while preserving indices
    det_list = [(k,v) for d in detections for k,v in d.items()]
    gt_list = [(k,v) for g in gt_boxes for k,v in g.items()]
    
    if not det_list or not gt_list:
        return [], 0, 0, 0
    
    # Cost matrix (1 - GIoU)
    cost_matrix = np.zeros((len(det_list), len(gt_list)))
    for i, (_, d_box) in enumerate(det_list):
        try:
            d_box = [float(x) for x in d_box]
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid detection box coordinates: {d_box}")
            continue
            
        for j, (_, gt_box) in enumerate(gt_list):
            try:
                gt_box = [float(x) for x in gt_box]
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid ground truth box coordinates: {gt_box}")
                continue
                
            try:
                giou = calculate_giou(d_box, gt_box)
                cost_matrix[i,j] = 1 - giou  # Convert to cost
            except Exception as e:
                logging.error(f"Error calculating GIoU for boxes {d_box} and {gt_box}: {str(e)}")
                cost_matrix[i,j] = 1  # Set maximum cost for invalid pairs
            
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter matches by GIoU threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        giou = 1 - cost_matrix[i,j]
        if giou >= giou_threshold:
            matches.append({
                'detection': det_list[i],
                'matched_gt': gt_list[j],
                'giou': giou
            })

    detection_P = len(matches) / len(det_list)
    detection_R = len(matches) / len(gt_list)
    mean_giou = np.mean([match['giou'] for match in matches])
    
    return matches, detection_P, detection_R, mean_giou


        