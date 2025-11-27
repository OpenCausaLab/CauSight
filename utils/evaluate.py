from .utils import match_detections_to_gt
from .vllm_infer import generate
from .prompt import *
from .img_server import process_image_path
from .utils import get_gt_pairs, extract_content
import ast
import json
import logging

def evaluate(entities, gt_pairs, predicted_pairs):
    """
    评估预测的因果关系对的准确性
    
    Args:
        entities: 实体列表，每个实体是一个字典 {name: bbox}
        gt_pairs: 真实的因果关系对列表
        predicted_pairs: 预测的因果关系对列表，每个对是一个字典 {cause: bbox, effect: bbox}
    
    Returns:
        tuple: (accuracy, f1, reward)
    """
    if not predicted_pairs:
        logging.warning("No predicted pairs provided")
        return 0, 0, 0, 0, 0, 0, 0
        
    if not entities:
        logging.warning("No entities provided")
        return 0, 0, 0, 0, 0, 0, 0
        
    if not gt_pairs:
        logging.warning("No ground truth pairs provided")
        return 0, 0, 0, 0, 0, 0, 0

    predicted_entities = []
    for pair in predicted_pairs:
        try:
            for key, value in pair.items():
                e = {key: value}
                if e not in predicted_entities:
                    predicted_entities.append(e)
        except (AttributeError, TypeError) as e:
            logging.error(f"Invalid pair format: {pair}")
            continue

    try:
        matches, detection_P, detection_R, mean_giou = match_detections_to_gt(predicted_entities, entities)
    except Exception as e:
        logging.error(f"Error matching detections to ground truth: {str(e)}")
        return 0, 0, 0, 0, 0, 0, 0

    table = []
    for match in matches:
        try:
            gt = match['matched_gt']
            ele = {gt[0]: gt[1]}
            index = entities.index(ele) + 1
            table.append({
                'entity': match['detection'][0],
                'bbox': match['detection'][1],
                'index': index,
                'giou': match['giou']
            })
        except (KeyError, ValueError, IndexError) as e:
            logging.error(f"Error processing match: {match}, Error: {str(e)}")
            continue

    predicted_relations = []
    for pair in predicted_pairs:
        try:
            relation = []
            for key, value in pair.items():
                try:
                    match = next(item for item in table if item['entity'] == key and item['bbox'] == value)
                    index, giou = match['index'], match['giou']
                    relation.append({'index': index, 'giou': giou})
                except StopIteration:
                    continue
                except (KeyError, TypeError) as e:
                    logging.error(f"Error processing relation for pair {pair}: {str(e)}")
                    continue

            if len(relation) == 2:    
                predicted_relations.append(relation)
        except (AttributeError, TypeError) as e:
            logging.error(f"Invalid pair format: {pair}")
            continue

    # count = 0
    # for relation in predicted_relations:
    #     try:
    #         r1 = relation[0]['index']
    #         r2 = relation[1]['index']
    #         if [r1, r2] in gt_pairs:
    #             count += 1
    #     except (IndexError, KeyError) as e:
    #         logging.error(f"Error checking relation {relation}: {str(e)}")
    #         continue

    unique_relations = set()
    for relation in predicted_relations:
        try:
            r1, r2 = relation[0]['index'], relation[1]['index']
            unique_relations.add((r1, r2))  # set 会自动去重
        except (IndexError, KeyError) as e:
            logging.error(f"Error checking relation {relation}: {str(e)}")
            continue

    count = sum(1 for (r1, r2) in unique_relations if [r1, r2] in gt_pairs)

    if len(predicted_pairs) == 0 or len(gt_pairs) == 0 or count == 0:
        return 0, 0, 0, 0, 0, 0, 0

    causal_P = count / len(predicted_pairs)
    causal_R = count / len(gt_pairs)

    predicted_id = set()
    for ele in predicted_relations:
        try:
            r1, r2 = ele[0]['index'], ele[1]['index']
            predicted_id.add(r1)
            predicted_id.add(r2)
        except (IndexError, KeyError) as e:
            logging.error(f"Error checking relation {ele}: {str(e)}")
            continue

    reachable_count = 0
    for ele in gt_pairs:
        try:
            r1, r2 = ele[0], ele[1]
            if r1 in predicted_id and r2 in predicted_id:
                reachable_count += 1
        except (IndexError, KeyError) as e:
            logging.error(f"Error checking relation {ele}: {str(e)}")
            continue
    
    ideal_P = reachable_count / len(predicted_pairs)
    ideal_R = reachable_count / len(gt_pairs)
    
    return causal_P, causal_R, detection_P, detection_R, mean_giou, ideal_P, ideal_R

def vanilla_inference(image_path, image_server, data):
    image_url_result = process_image_path(image_server, image_path)
    
    # Ensure we have a single URL string
    if isinstance(image_url_result, list):
        if len(image_url_result) > 0:
            image_url = image_url_result[0]  # Take first URL if it's a list
        else:
            print("No image URLs returned")
            return 0, 0, 0, 0, 0, 0, 0, "No image URLs returned"
    else:
        image_url = image_url_result

    gt_entities, gt_pairs = get_gt_pairs(data)

    result = generate(image_url=image_url, prompt=General_prompt)
    
    # Handle case where generate returns None
    if result is None or len(result) == 0:
        print("Generate function returned None or empty result")
        return 0, 0, 0, 0, 0, 0, 0, "No result generated"

    causal_pairs_text = extract_content(mark="causal pairs", text=result[0])
    
    # Handle case where extract_content returns None
    if causal_pairs_text is None:
        causal_pairs = []
        print(f"No causal pairs found in text: {result[0][:200]}...")
    else:
        try:
            causal_pairs = ast.literal_eval(causal_pairs_text)
        except (ValueError, SyntaxError):
            try:
                causal_pairs = json.loads(causal_pairs_text)
            except json.JSONDecodeError:
                causal_pairs = []
                print(f"Failed to parse causal pairs: {causal_pairs_text}")

    causal_P, causal_R, detection_P, detection_R, mean_giou, ideal_P, ideal_R = evaluate(gt_entities, gt_pairs, causal_pairs)

    return causal_P, causal_R, detection_P, detection_R, mean_giou, ideal_P, ideal_R, result[0]