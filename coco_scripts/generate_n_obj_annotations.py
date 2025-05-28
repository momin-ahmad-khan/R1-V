import json
from pycocotools.coco import COCO

# Load the original COCO annotations file
with open("/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/annotations/instances_val2017.json", "r") as f:
    coco_data = json.load(f)

print(len(coco_data['annotations']))

# Count the number of annotations per image
image_annotation_counts = {}
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    image_annotation_counts[image_id] = image_annotation_counts.get(image_id, 0) + 1

# Filter images that have exactly one annotation
single_object_image_ids = {img_id for img_id, count in image_annotation_counts.items() if count == 5}
print(len(single_object_image_ids))

# Filter images and annotations
filtered_images = [img for img in coco_data["images"] if img["id"] in single_object_image_ids]
filtered_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in single_object_image_ids]

# Create the new dataset dictionary
filtered_coco_data = {
    "info": coco_data["info"],
    "licenses": coco_data["licenses"],
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": coco_data["categories"]
}

# Save the filtered dataset
with open("instances_val2017_five_objects.json", "w") as f:
    json.dump(filtered_coco_data, f, indent=4)

print(f"Filtered dataset saved as instances_val2017_five_objects.json with {len(filtered_images)} images.")

