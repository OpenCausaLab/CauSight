from utils.utils import extract_content, restore_bbox
import ast
import json
import logging
import copy

class TreeNode:
    def __init__(self):
        self.action = 'SelectRegion'
        self.state = {}
        self.parent = None
        self.children = []
        self.visit_count = 0
        self.value = 0
        self.depth = 0
        self.is_fully_expanded = False
        self.is_terminal = False
        self.crop_info = None

    def initialize_state(self, last_node, result, crop_info):
        if last_node.parent is None: # root node
            try:
                description = extract_content('description', result) or ""
                think = extract_content('think', result) or ""
                region = extract_content('region name', result)
                bbox = extract_content('bounding box', result)
                
                if region is None or bbox is None:
                    logging.error(f"Missing required content in result: {result}")
                    raise ValueError("Missing region name or bounding box in result")
                
            except Exception as e:
                logging.error(f"Error extracting content: {str(e)}")
                raise
            self.state = {
                'trajectory': f"{description}\n{think}\nSo I need to focus on the \"{region}\" region, and the bounding box is {bbox}.\n\n",
                'explored_regions': [{'region_name': region, 'bounding_box': bbox}],
                'current_region': (region, bbox),
                'causal_pairs': [],
                'candidate_pairs': [],
            }
        else:
            inherited_state = copy.deepcopy(last_node.state)
            self.state = inherited_state
            self.crop_info = last_node.crop_info
            match last_node.action:
                case 'SelectRegion':
                    try:
                        think = extract_content('think', result) or ""
                        region = extract_content('region name', result)
                        bbox = extract_content('bounding box', result)
                        
                        if region is None or bbox is None:
                            logging.error(f"Missing required content in result: {result}")
                            raise ValueError("Missing region name or bounding box in result")
                            
                    except Exception as e:
                        logging.error(f"Error extracting content: {str(e)}")
                        raise
                    self.state['trajectory'] += f"{think}\nSo I need to focus on the \"{region}\" region, and the bounding box is {bbox}.\n\n"
                    if (region, bbox) not in self.state['explored_regions']:
                        self.state['explored_regions'].append({'region_name': region, 'bounding_box': bbox})
                    self.state['current_region'] = (region, bbox)
                case 'ProposePair':
                    try:
                        pairs = extract_content('entity pairs', result)
                        if pairs is None:
                            logging.warning(f"No entity pairs found in result: {result}")
                            pairs = "[]"  # 使用空列表作为默认值
                    except Exception as e:
                        logging.error(f"Error extracting content: {str(e)}")
                        raise
                    try:
                        pairs = ast.literal_eval(pairs)
                    except (ValueError, SyntaxError):
                        try:
                            pairs = json.loads(pairs)
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse pairs: {pairs}")
                            pairs = []
                    for p in pairs:
                        if p not in self.state['candidate_pairs']:
                            self.state['candidate_pairs'].append(p)
                    if crop_info is not None:
                        self.crop_info = crop_info
                        restore_pairs = []
                        for p in pairs:
                            try:
                                p_copy = copy.deepcopy(p)
                                p_copy[list(p_copy.keys())[0]] = restore_bbox(p_copy[list(p_copy.keys())[0]], crop_info)
                                p_copy[list(p_copy.keys())[1]] = restore_bbox(p_copy[list(p_copy.keys())[1]], crop_info)
                                restore_pairs.append(p_copy)
                            except Exception as e:
                                logging.warning(f"不正确的pair格式需要舍弃: {str(e)}")
                                continue
                    else:
                        raise ValueError("Crop info is not set")

                    self.state['trajectory'] += f"By Observation, this region contains the following correlated entity pairs: {str(restore_pairs)}.\n\n"
                case 'JudgeCausality':
                    try:
                        think = extract_content('think', result) or ""
                        pairs = extract_content('causal pairs', result)
                        if pairs is None:
                            logging.warning(f"No causal pairs found in result: {result}")
                            pairs = "[]"  # 使用空列表作为默认值
                    except Exception as e:
                        logging.error(f"Error extracting content: {str(e)}")
                        raise

                    try:
                        pairs = ast.literal_eval(pairs)
                    except (ValueError, SyntaxError):
                        try:
                            pairs = json.loads(pairs)
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse pairs: {pairs}")
                            pairs = []

                    if self.crop_info is not None and crop_info is None:
                        restore_pairs = []
                        for p in pairs:
                            try:
                                p_copy = copy.deepcopy(p)
                                p_copy[list(p_copy.keys())[0]] = restore_bbox(p_copy[list(p_copy.keys())[0]], self.crop_info)
                                p_copy[list(p_copy.keys())[1]] = restore_bbox(p_copy[list(p_copy.keys())[1]], self.crop_info)
                                restore_pairs.append(p_copy)
                            except Exception as e:
                                logging.warning(f"不正确的pair格式需要舍弃: {str(e)}")
                                continue
                    else:
                        raise ValueError("Crop info error")

                    for p in restore_pairs:
                        if p not in self.state['causal_pairs']:
                            self.state['causal_pairs'].append(p)

                    self.state['trajectory'] += f"{think}\nSo the entity pairs with causal relationships are {str(restore_pairs)}.\n\n"
                case _:
                    raise ValueError(f"Invalid action: {last_node.action}")

    def append_children(self, node):
        node.parent = self
        self.children.append(node)
        match self.action:
            case 'SelectRegion':
                node.action = 'ProposePair'
            case 'ProposePair':
                node.action = 'JudgeCausality'
            case 'JudgeCausality':
                node.action = 'SelectRegion'
            case _:
                raise ValueError(f"Invalid action when appending children: {self.action}")
        node.depth = self.depth + 1

    def update_value(self, value):
        self.value = value