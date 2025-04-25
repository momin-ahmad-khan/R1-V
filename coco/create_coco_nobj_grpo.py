from datasets import Dataset, Features, Value, Image as HFImage
from pycocotools.coco import COCO
from collections import defaultdict
import os
import json
from tqdm import tqdm

def create_grpo_ready_dataset(images_dir, annotations_file, n_obj=None):
    coco = COCO(annotations_file)
    dataset_entries = []

    img_ids = coco.getImgIds()
    for img_dict in tqdm(coco.loadImgs(img_ids), desc=f"Building GRPO dataset (n_obj={n_obj})"):
        img_id = img_dict['id']
        file_name = img_dict['file_name']
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Skip images without annotations
        if not annotations:
            continue

        if n_obj is not None and len(annotations) != n_obj:
            continue  # Skip if not exactly n_obj objects

        gt_dict = defaultdict(list)
        present_categories = set()

        for ann in annotations:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            gt_dict[cat_name].append([x1, y1, x2, y2])
            present_categories.add(cat_name)

        # Skip images without labeled categories
        if not present_categories:
            continue

        categories_str = ', '.join(sorted(present_categories))
        prompt = (
            f"Detect all {categories_str} in the image and return their locations in the form of coordinates. "
            "For each object category, provide a dictionary where keys are category names and values are lists of "
            "bounding boxes in the format [xmin, ymin, xmax, ymax]. "
            "Example format: {'category1': [[x1,y1,x2,y2], [x1,y1,x2,y2]], 'category2': [[x1,y1,x2,y2]]}. "
            "Only respond with the dictionary, no additional text."
        )

        dataset_entries.append({
            'image': img_path,
            'prompt': prompt,
            'solution': json.dumps(dict(gt_dict))
        })

    features = Features({
        'image': HFImage(),
        'prompt': Value('string'),
        'solution': Value('string')
    })

    return Dataset.from_list(dataset_entries, features=features)


n_obj = 3
dataset = create_grpo_ready_dataset(
    images_dir='/projects/EEHPC-DEV-2024D11-005/momin/datasets/coco/train2017',
    annotations_file='/projects/EEHPC-DEV-2024D11-005/momin/datasets/coco/annotations/instances_train2017.json',
    n_obj=n_obj
)

dataset_path = f'qwen_coco_grpo_iou_{n_obj}obj'
dataset.save_to_disk(dataset_path)

from datasets import load_from_disk
dataset = load_from_disk(dataset_path)
print(dataset[0])
