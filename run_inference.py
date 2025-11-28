from utils.prompt import *
from utils.evaluate import *
from utils.img_server import ImageServer
from utils.evaluate import evaluate, vanilla_inference

import logging
import json
import numpy as np

def get_data():
    with open("VCG-32K/COCO/annotations/test.jsonl", "r") as f:
        all_data = [json.loads(line) for line in f]
    return all_data

def main():

    image_server = ImageServer()
    image_server.start()

    all_data = get_data()
    causal_P_list = []
    causal_R_list = []
    detection_P_list = []
    detection_R_list = []
    mean_giou_list = []
    f1_list = []
    ideal_P_list = []
    ideal_R_list = []

    for data in all_data:
        try:
            image_path = data['images'][0]['image']
            image_path = f"VCG-32K/{image_path}"
        except:
            logging.error("error in finding image path")
            continue

        causal_P, causal_R, detection_P, detection_R, mean_giou, ideal_P, ideal_R, result = vanilla_inference(image_path, image_server, data)
        f1 = 2 * causal_P * causal_R / (causal_P + causal_R + 1e-10)

        causal_P_list.append(causal_P)
        causal_R_list.append(causal_R)
        detection_P_list.append(detection_P)
        detection_R_list.append(detection_R)
        mean_giou_list.append(mean_giou)
        f1_list.append(f1)
        ideal_P_list.append(ideal_P)
        ideal_R_list.append(ideal_R)
        with open("output/results.jsonl", "a") as f:
            f.write(json.dumps({"causal_P": causal_P, "causal_R": causal_R, "detection_P": detection_P, "detection_R": detection_R, "mean_giou": mean_giou, "f1": f1, "ideal_P": ideal_P, "ideal_R": ideal_R, "result": result}) + "\n")

    print(f"causal_P: {np.mean(causal_P_list)}, causal_R: {np.mean(causal_R_list)}, detection_P: {np.mean(detection_P_list)}, detection_R: {np.mean(detection_R_list)}, mean_giou: {np.mean(mean_giou_list)}, f1: {np.mean(f1_list)}, ideal_P: {np.mean(ideal_P_list)}, ideal_R: {np.mean(ideal_R_list)}, causal_P_std: {np.std(causal_P_list)}, causal_R_std: {np.std(causal_R_list)}, detection_P_std: {np.std(detection_P_list)}, detection_R_std: {np.std(detection_R_list)}, mean_giou_std: {np.std(mean_giou_list)}, f1_std: {np.std(f1_list)}, ideal_P_std: {np.std(ideal_P_list)}, ideal_R_std: {np.std(ideal_R_list)}")

if __name__ == "__main__":
    main()

