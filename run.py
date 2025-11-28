import json
import logging
import os
from datetime import datetime
import shutil

import numpy as np
from tqdm import tqdm   

from task import MCTSTask
from utils.img_server import ImageServer
from utils.utils import get_gt_pairs
from utils.evaluate import evaluate, vanilla_inference

def get_data():
    with open("VCG-32K/COCO/annotations/train.jsonl", "r") as f:
        all_data = [json.loads(line) for line in f]
        all_data = all_data[0:100]
    return all_data

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log_gpu0"),
            logging.StreamHandler(),
        ],
    )

def main():
    setup_logging()
    logging.info("Starting the program")

    #start image server
    image_server = ImageServer()
    image_server.start()

    all_data = get_data()

    for data in all_data:
        try:
            image_path = data['images'][0]['image']
            id = image_path.split('train/')[-1].split('.')[0]
            image_path = f"VCG-32K/{image_path}"
        except:
            logging.error("error in finding image path")
            continue

        os.makedirs(f"temp/{id}", exist_ok=True)
        task = MCTSTask(data=data, data_idx=id, image_path=image_path, image_server=image_server)

        root_node, search_metric = task.run()
        best_leaf_node = task.get_best_path(root_node)
        
        gt_entities, gt_pairs = get_gt_pairs(data)
        predicted_pairs = best_leaf_node.state['causal_pairs']
        causal_P, causal_R, _, _, _, _, _ = evaluate(gt_entities, gt_pairs, predicted_pairs)

        v_causal_P, v_causal_R, _, _, _, _, _, v_result = vanilla_inference(image_path, image_server, data)

        best_leaf_node.state['precision'] = causal_P
        best_leaf_node.state['recall'] = causal_R
        best_leaf_node.state['vanilla_precision'] = v_causal_P
        best_leaf_node.state['vanilla_recall'] = v_causal_R
        best_leaf_node.state['vanilla_result'] = v_result

        best_leaf_node.state['image_id'] = id
        best_leaf_node.state['image_path'] = image_path
        best_leaf_node.state["search_metric"] = search_metric

        with open(f"ToCT/raw_sft_data.jsonl", "a") as f:
            json.dump(best_leaf_node.state, f)
            f.write("\n")

        if causal_R != 0 or v_causal_R != 0:
            if v_causal_R >= causal_R:
                sft = {
                    "image_id": id,
                    "image_path": image_path,
                    "trajectory": v_result
                }
            else:
                sft = {
                    "image_id": id,
                    "image_path": image_path,
                    "trajectory": f"{best_leaf_node.state['trajectory']}<'causal pairs'>\n{str(best_leaf_node.state['causal_pairs'])}\n</causal pairs>"
                }
            with open(f"ToCT/sft_data.jsonl", "a") as f:
                json.dump(sft, f)
                f.write("\n")

        shutil.rmtree(f"temp/{id}")

    image_server.stop()

if __name__ == "__main__":
    main()



