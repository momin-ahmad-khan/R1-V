from datasets import Dataset, Features, Value, Image as HFImage
from pycocotools.coco import COCO
from collections import defaultdict
import os
import json
from tqdm import tqdm

def create_grpo_ready_dataset(images_dir, annotations_file, n_obj=None, save_dir_root="qwen_coco_grpo"):
    coco = COCO(annotations_file)
    dataset_entries = []
    matched_count = 0

    img_ids = coco.getImgIds()

    for img_dict in tqdm(coco.loadImgs(img_ids), desc=f"Building GRPO dataset (n_obj={n_obj})"):
        img_id = img_dict['id']
        file_name = img_dict['file_name']
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        if not annotations:
            continue

        if n_obj is not None and len(annotations) != n_obj:
            continue

        gt_dict = defaultdict(list)
        present_categories = set()

        for ann in annotations:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            gt_dict[cat_name].append([x1, y1, x2, y2])
            present_categories.add(cat_name)

        if not present_categories:
            continue

        categories_str = ', '.join(sorted(present_categories))
        prompt = (
            f"Please output bbox coordinates and names of {categories_str}."
        )

        dataset_entries.append({
            'image': img_path,
            'prompt': prompt,
            'solution': json.dumps(dict(gt_dict))
        })

        matched_count += 1

    print(f"\n✅ Found {matched_count} images with exactly {n_obj} object{'s' if n_obj != 1 else ''}.")

    features = Features({
        'image': HFImage(),
        'prompt': Value('string'),
        'solution': Value('string')
    })

    dataset = Dataset.from_list(dataset_entries, features=features)

    # Save the dataset with a name reflecting the number of objects
    save_dir = f"{save_dir_root}_nobj{n_obj}" if n_obj is not None else save_dir_root
    dataset.save_to_disk(save_dir)

    print(f"✅ Dataset saved at: {save_dir}")

    return dataset

dataset = create_grpo_ready_dataset(
    images_dir='/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/train2017',
    annotations_file='/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/annotations/instances_train2017.json',
    n_obj=5,  # <-- You control how many objects!
    save_dir_root="qwen_coco_grpo"
)

