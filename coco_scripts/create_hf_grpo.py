from datasets import Dataset, Features, Value, Sequence, Image as HFImage
from pycocotools.coco import COCO
from collections import defaultdict
import os
import json

def create_grpo_ready_dataset(images_dir, annotations_file):
    coco = COCO(annotations_file)
    dataset_entries = []

    for img_dict in coco.loadImgs(coco.getImgIds()):
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
            'solution': json.dumps(dict(gt_dict))  # Stringify for eval() during training
        })

    features = Features({
        'image': HFImage(),
        'prompt': Value('string'),
        'solution': Value('string')  # <-- GRPO will use this during reward computation
    })

    return Dataset.from_list(dataset_entries, features=features)

# Example usage
dataset = create_grpo_ready_dataset(
    images_dir='/ceph/hpc/home/eumomink/datasets/coco/val2017',
    annotations_file='/ceph/hpc/home/eumomink/datasets/coco/annotations/instances_val2017.json'
)

dataset.save_to_disk('qwen_coco_grpo_iou')


from datasets import load_from_disk
dataset = load_from_disk('qwen_coco_grpo_iou')
print(dataset[0])
