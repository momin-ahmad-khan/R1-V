# Re-run after code execution environment reset

import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Configuration
nobj=7
results_dir = "/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/results_7obj_fiomcrR"
annotation_path = "/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/annotations/instances_val2017_seven_objects.json"

# Checkpoint range
checkpoints = list(range(50, 650, 50))
map_scores = []

def parse_bbox(bbox_entry):
    def flatten(lst):
        for item in lst:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    try:
        elements = []
        for item in flatten(bbox_entry):
            if isinstance(item, str):
                elements.extend(item.split(','))
            else:
                elements.append(str(item))

        numbers = [float(x.strip()) for x in elements if x.strip()]
        if len(numbers) != 4 or any(n < 0 for n in numbers):
            return None
        xmin, ymin, xmax, ymax = numbers
        if xmin >= xmax or ymin >= ymax:
            return None
        return [xmin, ymin, xmax, ymax]
    except:
        return None

for ckpt in checkpoints:
    pred_path = os.path.join(results_dir, f"vlm_coco_results_grpo_{nobj}obj_chkpt{ckpt}.json")
    if not os.path.exists(pred_path):
        map_scores.append((ckpt, None))
        continue

    with open(pred_path, "r") as f:
        predictions_data = json.load(f)

    coco = COCO(annotation_path)
    categories = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}

    coco_predictions = []
    for image_id, data in predictions_data.items():
        image_id = int(image_id)
        detections = data["detections"][0] if isinstance(data["detections"], list) else data["detections"]
        for category_name, bboxes in detections.items():
            if category_name not in categories:
                continue

            category_id = categories[category_name]
            if isinstance(bboxes, list) and all(isinstance(x, (int, float)) for x in bboxes):
                bboxes = [bboxes]
            elif not isinstance(bboxes, list):
                bboxes = [bboxes]

            for bbox_entry in bboxes:
                parsed_bbox = parse_bbox(bbox_entry)
                if not parsed_bbox:
                    continue
                xmin, ymin, xmax, ymax = parsed_bbox
                width, height = xmax - xmin, ymax - ymin
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, width, height],
                    "score": 1.0
                })

    formatted_predictions_path = f"formatted_predictions_{ckpt}.json"
    with open(formatted_predictions_path, "w") as f:
        json.dump(coco_predictions, f)

    coco_dt = coco.loadRes(formatted_predictions_path)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    #mean average precision
    map_scores.append((ckpt, coco_eval.stats[0]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# map_scores[11] = map_scores[10]
df = pd.DataFrame(map_scores, columns=["Checkpoint", "mAP"])
df = df.dropna()

plt.figure()
sns.lineplot(data=df, x="Checkpoint", y="mAP", marker="o")
# plt.axhline(y=0.460, color='red', linestyle='--', label='Baseline (0.460)')
# plt.axhline(y=0.396, color='red', linestyle='--', label='Baseline (0.396)')
plt.axhline(y=0.315, color='red', linestyle='--', label='Baseline (0.315)')
plt.title("mAP, 7obj, iou + format + object coverage + missing + classification + redundancy + Recall")
plt.xlabel("Checkpoint")
plt.ylabel("Mean Average Precision (mAP)")
plt.grid(True)
plt.tight_layout()
plt.savefig("map_7obj_fiomcrR.png")
