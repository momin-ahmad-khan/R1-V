import re
import ast
import numpy as np
from sklearn.metrics import f1_score, recall_score

def parse_qwen_detection_output(output_str):
    try:
        match = re.search(r"\[\s*{.*?}\s*]", output_str, re.DOTALL)
        if not match:
            return []
        content = match.group(0)
        return ast.literal_eval(content)
    except Exception:
        return []

def group_boxes_by_label(pred_list):
    grouped = {}
    for item in pred_list:
        label = item.get("label")
        bbox = item.get("bbox_2d")
        if label and isinstance(bbox, list) and len(bbox) == 4:
            grouped.setdefault(label, []).append(bbox)
    return grouped

def extract_gt_dict(gt_str):
    try:
        return ast.literal_eval(gt_str)
    except Exception:
        return {}

def missing_object_reward(completions, solution, **kwargs):
    rewards = []
    for comp, sol in zip(completions, solution):
        pred = group_boxes_by_label(parse_qwen_detection_output(comp[0]["content"]))
        gt = extract_gt_dict(sol)

        reward = 0.0
        for label in gt:
            if label in pred:
                reward += min(len(pred[label]), len(gt[label])) / len(gt[label])
        reward /= len(gt) if gt else 1
        rewards.append(reward)
    return rewards

def classification_accuracy_reward(completions, solution, **kwargs):
    rewards = []
    for comp, sol in zip(completions, solution):
        pred = group_boxes_by_label(parse_qwen_detection_output(comp[0]["content"]))
        gt = extract_gt_dict(sol)

        pred_labels = list(pred.keys())
        gt_labels = list(gt.keys())

        correct = sum([1 for label in pred_labels if label in gt_labels])
        acc = correct / len(gt_labels) if gt_labels else 1.0
        rewards.append(acc)
    return rewards

def redundancy_penalty(completions, solution, **kwargs):
    rewards = []
    for comp, sol in zip(completions, solution):
        pred = group_boxes_by_label(parse_qwen_detection_output(comp[0]["content"]))
        gt = extract_gt_dict(sol)

        penalty = 0.0
        for label in gt:
            pred_count = len(pred.get(label, []))
            gt_count = len(gt[label])
            if pred_count > gt_count:
                penalty += (pred_count - gt_count) / (gt_count + 1e-5)
        rewards.append(-penalty)
    return rewards

def hallucination_penalty(completions, solution, **kwargs):
    rewards = []
    for comp, sol in zip(completions, solution):
        pred = group_boxes_by_label(parse_qwen_detection_output(comp[0]["content"]))
        gt = extract_gt_dict(sol)

        hallucinated_labels = set(pred.keys()) - set(gt.keys())
        rewards.append(-len(hallucinated_labels))
    return rewards

def recall_f1_reward(completions, solution, **kwargs):
    rewards = []
    for comp, sol in zip(completions, solution):
        pred = group_boxes_by_label(parse_qwen_detection_output(comp[0]["content"]))
        gt = extract_gt_dict(sol)

        all_labels = list(set(list(pred.keys()) + list(gt.keys())))
        y_true = [1 if label in gt else 0 for label in all_labels]
        y_pred = [1 if label in pred else 0 for label in all_labels]

        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # rewards.append({"f1": f1, "recall": recall})
        rewards.append(recall)
    return rewards
