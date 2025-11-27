Caption_prompt = """
Your task is to analyze the causal relationships between entities in the image through multiple steps.
At the current step, your need to briefly describe the image, then think and select a main region to focus on, and output the name and bounding box of that region.

Your output format should be as follows:

<description>
[Describe the image as concisely as possible.]
</description>

<think>
[Provide the concise thinking process for selecting the focused region.]
</think>

<region name>
[Output the name of the focused region.]
</region name>

<bounding box>
[Output the bounding box of the focused region with format [x1, y1, x2, y2] and nothing else, where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate of the bounding box.]
</bounding box>
"""

SelectRegion_prompt = """
You are analyzing the causal relationships between entities in the image through multiple steps.
Your current reasoning trajectory is as follows:

Explored regions: {explored_regions}.

Identified causal pairs: {causal_pairs}.

Now we hope to look for new regions to discover more potential correlated entity pairs.
Please select the next most worthy region to focus on and explain your thinking process.
Note: the next region should be DIFFERENT from the previous explored regions.

- If you think the exploration regions and identified causal pairs are SUFFICIENTLY COMPREHENSIVE, you should **DIRECTLY** output "END TRACE" and nothing else.

Otherwise, your output format should be as follows:

<think>
[State the reason as concisely as possible for selecting the new focused region.]
</think>

<region name>
[Output the name of the focused region and nothing else.]
</region name>

<bounding box>
[Output the bounding box of the focused region with format [x1, y1, x2, y2] and nothing else, where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate of the bounding box.]
</bounding box>
"""


ProposePair_prompt = """
Your task is to identify all entity pairs that may have correlations in the image.
Each pair should have obvious potential correlations such as spatial dependence, support, grasping, placement, inclusion, etc.
Think and output all these correlated entity pairs and their bounding boxes.

Your output format should be as follows:

<think>
[Provide the concise thinking process for identifying correlated entity pairs.]
</think>

<entity pairs>
[Output all the correlated entity pairs in the format of "[{\"entity1\": [x1, y1, x2, y2], \"entity2\": [x1, y1, x2, y2]}, {\"entity3\": [x1, y1, x2, y2], \"entity4\": [x1, y1, x2, y2]}, ...]". You should use ACTUAL ENTITY NAME to replace the placeholders "entity1", "entity2", ... in the format. (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate of the bounding box.]
</entity pairs>
"""

JudgeCausality_prompt = """
Based on the image, your task is to determine whether causal relationships exist between the following entity pairs.
Entity pairs: {entity_pairs}

The causality criteria are as follows:
For example, if the entity pairs are {{"A": [x1, y1, x2, y2], "B": [x1, y1, x2, y2]}} or {{"B": [x1, y1, x2, y2], "A": [x1, y1, x2, y2]}}:
- A is in **direct contact** with B. 
- A's **presence** maintains B's **current state**.  
- Removing A would cause B to **lose its current state**.
Then A is the cause and B is the effect.
(x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate of the bounding box.

Your output format should be as follows:

<think>
[Consider entity pairs and keep the reasoning as concise as possible.]
</think>

<causal pairs>
[Output entity pairs with causal relationships only and if necessary, swap the ORDER of entities pairs to ensure the cause precedes the effect.]
</causal pairs>
"""

General_prompt = """Identify all **causal relationships** between entities in the image based on the following criteria:
- A is in **direct contact** with B.
- A's **presence** maintains B's **current state**.
- Removing A would cause B to **lose its current state**.
Then A is the cause and B is the effect.

Please provide your reasoning process and output all the entity pairs with causal relationships and their bounding boxes in the following format:

<think>
[Provide your reasoning process for analyzing the image.]
</think>

<causal pairs>
[Output all the entity pairs with causal relationships and their bounding boxes in the format of "[{\"cause\": [x1, y1, x2, y2], \"effect\": [x1, y1, x2, y2]}, {\"cause\": [x1, y1, x2, y2], \"effect\": [x1, y1, x2, y2]}, ...]". You should use ACTUAL ENTITY NAME to replace the placeholders "cause" and "effect" in the format. (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate of the bounding box.]
</causal pairs>
"""





