import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import string
import os
import uuid
import tempfile

from utils.img_server import process_image_path
from utils.vllm_infer import generate
from utils.prompt import *
from utils.utils import zoom_in, get_gt_pairs, extract_content, match_detections_to_gt
from utils.evaluate import evaluate

from node import TreeNode
from search import mcts_entrance, execute_round

import logging

import numpy as np


class MCTSTask:
    def __init__(
        self, 
        iteration_limit=20,  
        exploration_constant=1.0,
        low_value=0,
        image_path=None,
        image_server=None,
        data=None,
        data_idx=None,
        alpha=0.3,
        max_regions=4,
        max_pairs=20
    ):
        # Task parameters
        self.alpha = alpha
        self.iteration_limit = iteration_limit
        self.data = data
        self.data_idx = data_idx
        self.image_path = image_path
        self.image_server = image_server
        self.low_value = low_value
        self.exploration_constant = exploration_constant
        self.image_url = process_image_path(self.image_server, self.image_path)
        self.temp_image_path = None
        self.temp_image_url = None
        self.root_node = None
        self.max_regions = max_regions
        self.max_pairs = max_pairs

    def step(self, current_node):
        """
        MCTS step.
        """
        crop_info = None

        if current_node.parent is None: # root node
            prompt = Caption_prompt
            results = generate(image_url=self.image_url, prompt=prompt)
            if results is None:
                logging.error("Failed to generate results for root node")
                return None
            proposed_sub_nodes = []
            for result in results:
                try:
                    sub_node = TreeNode()
                    sub_node.initialize_state(current_node, result, crop_info)
                    proposed_sub_nodes.append(sub_node)
                except Exception as e:
                    logging.error(f"Failed to initialize root sub_node: {str(e)}")
                    continue
            
            # 如果没有成功创建任何子节点，将当前节点标记为终端节点
            if not proposed_sub_nodes:
                current_node.is_terminal = True
                return None
                
            return proposed_sub_nodes
        else:
            explored_regions = current_node.state['explored_regions']
            causal_pairs = current_node.state['causal_pairs']
            candidate_pairs = current_node.state['candidate_pairs']

            match current_node.action:
                case 'SelectRegion':
                    if len(explored_regions) >= self.max_regions or len(causal_pairs) >= self.max_pairs:
                        current_node.is_terminal = True
                        return None
                    
                    prompt = SelectRegion_prompt.format(explored_regions=explored_regions, causal_pairs=causal_pairs)
                    results = generate(image_url=self.image_url, prompt=prompt)
                case 'ProposePair':
                    prompt = ProposePair_prompt
                    current_region = current_node.state['current_region']
                    bbox = current_region[1]
                    try:
                        idid = self.data_idx.split('.')[0] if '.' in self.data_idx else self.data_idx
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix='.jpg', 
                            delete=False,
                            dir=f'temp/{idid}'  # 指定目录
                        )
                        self.temp_image_path = temp_file.name
                        temp_file.close()

                        crop_info = zoom_in(image_path=self.image_path, bbox=bbox, output_path=self.temp_image_path)
                        self.temp_image_url = process_image_path(self.image_server, self.temp_image_path)
                    except Exception as e:
                        logging.error(f"Failed to crop image: {str(e)}")
                        current_node.is_terminal = True
                        return None
                    results = generate(image_url=self.temp_image_url, prompt=prompt)
                case 'JudgeCausality':
                    prompt = JudgeCausality_prompt.format(entity_pairs=candidate_pairs)
                    current_node.state['candidate_pairs'] = []
                    results = generate(image_url=self.temp_image_url, prompt=prompt)
                case _:
                    raise ValueError(f"Invalid action: {current_node.action}")
                
            if results is None:
                logging.error(f"Failed to generate results for action {current_node.action}")
                current_node.is_terminal = True
                return None
                
            proposed_sub_nodes = []
            for result in results:
                if "END TRACE" in result:
                    current_node.is_terminal = True
                    return None
                try:
                    sub_node = TreeNode()
                    sub_node.initialize_state(current_node, result, crop_info)
                    proposed_sub_nodes.append(sub_node)
                except Exception as e:
                    logging.error(f"Failed to initialize sub_node: {str(e)}")
                    continue
            
            # 如果没有成功创建任何子节点，将当前节点标记为终端节点
            if not proposed_sub_nodes:
                current_node.is_terminal = True
                return None
                
            return proposed_sub_nodes

    def reward(self, node):
        """
        Reward function.
        """
        entities, gt_pairs = get_gt_pairs(self.data)
        predicted_pairs = node.state['causal_pairs']
        
        # Handle case where gt_pairs is empty to prevent ZeroDivisionError
        if len(gt_pairs) == 0:
            length_reward = 0 if len(predicted_pairs) == 0 else -1  # Penalty for predicting when no ground truth exists
        else:
            length_reward = len(predicted_pairs) / len(gt_pairs)
            
        region_reward = len(node.state['explored_regions'])
        causal_P, causal_R, _, _, _, _, _ = evaluate(entities, gt_pairs, predicted_pairs)
        causal_reward = 0.75 * causal_R + 0.25 * causal_P + 0.05 * length_reward + 0.005 * region_reward
        return causal_reward

    def run(self):
        """
        Run MCTS search.

        Returns:
            TreeNode: Root node of the search tree
        """
        try:
            root_node, search_metric = mcts_entrance(self)
            self.root_node = root_node  # Store for class-level access if needed
            print(f"Search completed with {search_metric} seconds")
            return root_node, search_metric
        except Exception as e:
            logging.error(f"Error during MCTS search: {str(e)}")
            raise

    def get_best_path(self, node):
        """
        Get the leaf node with the highest value.

        Returns:
            TreeNode: Leaf node with the highest value
        """
        while not node.is_terminal and node.children:
            values = []
            for child_node in node.children:
                values.append(child_node.value)
            
            # Handle empty values list (shouldn't happen due to while condition, but safety check)
            if not values:
                break
                
            best_value = max(values)
            
            # Use a small epsilon for floating-point comparison to avoid precision issues
            epsilon = 1e-10
            best_child_nodes = [child_node for child_node in node.children 
                              if abs(child_node.value - best_value) < epsilon]
            
            # Safety check: if no children found due to precision issues, fall back to direct comparison
            if not best_child_nodes:
                best_child_nodes = [child_node for child_node in node.children if child_node.value == best_value]
            
            # Final safety check: if still no children, just pick the first one
            if not best_child_nodes:
                best_child_nodes = [node.children[0]]
            
            node = random.choice(best_child_nodes)
        return node

                