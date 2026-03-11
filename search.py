import math
import random
import time

import numpy

from node import TreeNode


def mcts_entrance(mcts_task):
    root_node = TreeNode()

    search_start_time = time.time()
    for iteration_count in range(mcts_task.iteration_limit):
        print(f"<Begin search round {iteration_count + 1}/{mcts_task.iteration_limit}>")
        root_node = execute_round(root_node, mcts_task)
    
    search_metric = time.time() - search_start_time

    return root_node, search_metric


def execute_round(root_node, mcts_task):
    selection_path = []
    
    print("*" * 30, "phase selection", "*" * 30, "\n")
    selected_node = select_node(root_node, mcts_task, selection_path)
    print(f"Selected node: {selected_node.action}, depth: {selected_node.depth}\n")

    print("*" * 30, "phase expansion", "*" * 30, "\n")
    simulation_start_node = selected_node
    outcome_reward = None
    
    if selected_node.is_terminal:
        print("This is a terminal node, no further expansion required.\n")
        outcome_reward = mcts_task.reward(selected_node)
    else:
        expanded_child = expand_node(selected_node, mcts_task)
        if expanded_child != selected_node:
            simulation_start_node = expanded_child
            selection_path.append(expanded_child)
            print(f"Complete expansion!, expanded node count: {len(selected_node.children)}")
            print(f"Selected child for simulation: {expanded_child.action}")
        else:
            print("Node marked as terminal during expansion.\n")
            outcome_reward = mcts_task.reward(selected_node)

    print("*" * 30, "phase simulation", "*" * 30, "\n")
    if outcome_reward is None:
        if simulation_start_node.is_terminal:
            outcome_reward = mcts_task.reward(simulation_start_node)
            print("Simulation start node is terminal, using terminal reward.\n")
        else:
            outcome_reward, rollout_path = simulate_node(simulation_start_node, mcts_task)
            selection_path.extend(rollout_path)

    print("*" * 30, "phase backpropagation", "*" * 30, "\n")
    back_propagate(selection_path, outcome_reward, mcts_task)

    return root_node


def select_node(current_node, mcts_task, selection_path):
    selection_path.append(current_node)
    while current_node.is_fully_expanded and not current_node.is_terminal:
        current_node = get_best_child(current_node, mcts_task)
        selection_path.append(current_node)
    return current_node


def get_best_child(parent_node, mcts_task):
    if not parent_node.children:
        parent_node.is_terminal = True
        return parent_node
    
    best_value = mcts_task.low_value
    best_child_nodes = []
    
    for child_node in parent_node.children:
        # UCB1 formula for node selection
        if child_node.visit_count > 0:
            exploitation_term = child_node.value
            exploration_term = mcts_task.exploration_constant * math.sqrt(
                2 * math.log(parent_node.visit_count) / child_node.visit_count
            )
            ucb_value = exploitation_term + exploration_term
        else:
            ucb_value = child_node.value + 1.0 

        if ucb_value > best_value:
            best_value = ucb_value
            best_child_nodes = [child_node]
        elif ucb_value == best_value:
            best_child_nodes.append(child_node)
    
    if not best_child_nodes:
        ucb_values = []
        for child_node in parent_node.children:
            if child_node.visit_count > 0:
                exploitation_term = child_node.value
                exploration_term = mcts_task.exploration_constant * math.sqrt(
                    2 * math.log(parent_node.visit_count) / child_node.visit_count
                )
                ucb_value = exploitation_term + exploration_term
            else:
                ucb_value = child_node.value + 1.0
            ucb_values.append(ucb_value)
        
        best_ucb_value = max(ucb_values)
        best_child_nodes = [child_node for i, child_node in enumerate(parent_node.children) 
                           if ucb_values[i] == best_ucb_value]
    
    return random.choice(best_child_nodes)


def expand_node(current_node, mcts_task):
    """
    扩展节点并返回一个新的子节点用于simulation
    """
    proposed_sub_nodes = mcts_task.step(current_node)
    if proposed_sub_nodes is None:
        current_node.is_terminal = True
        return current_node
    
    new_children = []
    for sub_node in proposed_sub_nodes:
        existing_states = [child.state for child in current_node.children]
        if sub_node.state not in existing_states:
            current_node.append_children(sub_node)
            new_children.append(sub_node)

    current_node.is_fully_expanded = True
    
    if new_children:
        return random.choice(new_children)
    else:
        if current_node.children:
            return random.choice(current_node.children)
        else:
            current_node.is_terminal = True
            return current_node

def simulate_node(current_node, mcts_task):
    """
    执行rollout并跟踪路径中的所有节点
    """
    rollout_path = []
    while not current_node.is_terminal:
        proposed_sub_nodes = mcts_task.step(current_node)
        if proposed_sub_nodes is None:
            current_node.is_terminal = True
            break
        
        proposed_node = random.choice(proposed_sub_nodes)
        current_node.append_children(proposed_node)
        rollout_path.append(proposed_node)
        current_node = proposed_node
    
    outcome_reward = mcts_task.reward(current_node)
    current_node.update_value(outcome_reward)
    return outcome_reward, rollout_path


def back_propagate(selection_path, outcome_reward, mcts_task):
    for node in reversed(selection_path):
        node.visit_count += 1
        if hasattr(mcts_task, 'alpha'):
            node.value = (
                node.value * (1 - mcts_task.alpha) + outcome_reward * mcts_task.alpha
            )
        else:
            node.value = ((node.value * (node.visit_count - 1)) + outcome_reward) / node.visit_count