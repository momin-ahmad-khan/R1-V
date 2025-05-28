import os
import re
import ast
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from rewards import *


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    print("#####################")
    print(completions)
    print("#####################")
    print(solution)
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]

def coco_to_xyxy(box):
    """Convert [x, y, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = box
    return [x, y, x + w, y + h]
    
def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def extract_dict_string(text):
    """Extract the first valid Python dict substring from the text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if inter_area == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou**2

def extract_and_group_boxes(text):
    text = text.replace("，", ",")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    raw_dict = match.group(0)

    # Convert duplicate keys into grouped lists using regex
    # Step 1: Find all key-value pairs
    key_vals = re.findall(r"'([^']+)'\s*:\s*\[.*?\]", raw_dict)
    # Step 2: Replace duplicates manually
    grouped = {}
    for m in re.finditer(r"'([^']+)'\s*:\s*(\[[^\[\]]+\])", raw_dict):
        key = m.group(1)
        val_str = m.group(2)
        val = ast.literal_eval(val_str)
        grouped.setdefault(key, []).append(val)
    return grouped

def iou_reward(completions, solution, **kwargs):
    # print("completion: ", completions)
    # print("solution: ", solution)
    rewards = []

    for i in range(len(completions)):
        try:
            raw_pred = completions[i][0]["content"]
            raw_gt = solution[i]

            pred = extract_and_group_boxes(raw_pred)
            gt = ast.literal_eval(raw_gt.replace("，", ","))

            if not isinstance(pred, dict) or not isinstance(gt, dict):
                raise ValueError("Parsed output not a dict")

            ious = []
            for category in gt:
                gt_boxes = gt[category]
                pred_boxes = pred.get(category, [])

                for gt_box in gt_boxes:
                    best_iou = 0
                    for pred_box in pred_boxes:
                        best_iou = max(best_iou, compute_iou(gt_box, pred_box))
                    ious.append(best_iou)

            rewards.append(np.mean(ious) if ious else 0.0)

        except Exception as e:
            print(f"[WARN] Failed to compute IoU reward: {e}")
            rewards.append(0.0)
    print(rewards)
    return rewards

# Define updated reward function
def iou_reward_qwen(completions, solution, **kwargs):
    rewards = []
    for i in range(len(completions)):
        try:
            raw_pred = completions[i][0]["content"]
            raw_gt = solution[i]

            # Strip markdown code block if present
            if raw_pred.startswith("```json"):
                raw_pred = raw_pred[7:].strip()
            if raw_pred.endswith("```"):
                raw_pred = raw_pred[:-3].strip()

            parsed_pred = json.loads(raw_pred)
            gt = ast.literal_eval(raw_gt.replace("，", ","))

            # Group predictions by label
            pred = {}
            for obj in parsed_pred:
                label = obj["label"]
                box = obj["bbox_2d"]
                pred.setdefault(label, []).append(box)

            if not isinstance(pred, dict) or not isinstance(gt, dict):
                raise ValueError("Parsed output not a dict")

            ious = []
            for category in gt:
                gt_boxes = gt[category]
                pred_boxes = pred.get(category, [])

                for gt_box in gt_boxes:
                    best_iou = 0
                    for pred_box in pred_boxes:
                        best_iou = max(best_iou, compute_iou(gt_box, pred_box))
                    ious.append(best_iou)

            rewards.append(np.mean(ious) if ious else 0.0)

        except Exception as e:
            print(f"[WARN] Failed to compute IoU reward: {e}")
            rewards.append(0.0)
    return rewards

def format_reward(completions, solution, **kwargs):
    """
    Reward function that checks if:
    - Completion is valid JSON
    - Each item is a dict with "bbox_2d" and "label"
    - All labels from the ground truth appear at least once in the predicted output
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for idx, content in enumerate(completion_contents):
        try:
            # Step 1: Clean and parse the completion
            content = content.strip()
            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.endswith("```"):
                content = content[:-len("```")].strip()

            parsed = json.loads(content)

            if not isinstance(parsed, list):
                raise ValueError("Not a list")

            for item in parsed:
                if not isinstance(item, dict) or "bbox_2d" not in item or "label" not in item:
                    raise ValueError("Malformed entry")

            # Step 2: Extract predicted labels
            pred_labels = set(item["label"] for item in parsed)

            # Step 3: Parse ground truth and extract required labels
            gt = json.loads(solution[idx]) if isinstance(solution[idx], str) else solution[idx]
            if isinstance(gt, dict):
                gt_labels = set(gt.keys())
            elif isinstance(gt, list):
                gt_labels = set(obj["label"] for obj in gt)
            else:
                raise ValueError("Unrecognized ground truth format")

            # Step 4: Penalize missing labels
            missing_labels = gt_labels - pred_labels
            reward = 1.0 if not missing_labels else 0.0

        except Exception as e:
            reward = 0.0

        rewards.append(reward)

    return rewards

def object_coverage_reward(completions, solution, **kwargs):
    """
    Reward function that penalizes missing object categories.
    Returns 1.0 if all ground truth labels are present in the completion, else 0.0.
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for idx, content in enumerate(completion_contents):
        try:
            # Strip JSON formatting
            content = content.strip()
            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.endswith("```"):
                content = content[:-len("```")].strip()

            # Parse predicted labels
            pred = json.loads(content)
            pred_labels = set(item["label"] for item in pred if isinstance(item, dict) and "label" in item)

            # Parse ground truth labels
            raw_gt = solution[idx]
            gt = json.loads(raw_gt) if isinstance(raw_gt, str) else raw_gt
            if isinstance(gt, dict):
                gt_labels = set(gt.keys())
            elif isinstance(gt, list):
                gt_labels = set(obj["label"] for obj in gt)
            else:
                raise ValueError("Unrecognized ground truth format")

            # Check label coverage
            missing_labels = gt_labels - pred_labels
            reward = 1.0 if not missing_labels else 0.0

        except Exception as e:
            reward = 0.0

        rewards.append(reward)

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "iou": iou_reward_qwen,
    "obj_coverage": object_coverage_reward,
    "missing": missing_object_reward,
    "classification": classification_accuracy_reward,
    "redundancy": redundancy_penalty,
    "hallucination": hallucination_penalty,
    "recall": recall_f1_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    #dataset = load_from_disk('/ceph/hpc/home/eumomink/datasets/coco/qwen_coco_grpo_iou')
    dataset = load_from_disk('/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/qwen_coco_grpo_nobj7')

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["prompt"]},
                    ],
                },
            ],
        }


    if "image" in dataset.features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
        print(dataset[0])
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

