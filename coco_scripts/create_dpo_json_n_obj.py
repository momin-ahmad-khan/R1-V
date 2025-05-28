import json
import os
from pycocotools.coco import COCO
from collections import defaultdict
from PIL import Image
import random
import math
from tqdm import tqdm

# --- Configuration ---
# !! IMPORTANT: Adjust these paths if they are different !!
COCO_BASE_DIR = "/ceph/hpc/home/eumomink/datasets/coco"  # Base directory where 'train2017' and 'annotations' folders reside
IMAGE_DIR = os.path.join(COCO_BASE_DIR, "train2017")
ANNOTATION_FILE = os.path.join(COCO_BASE_DIR, "annotations", "instances_train2017.json")
TARGET_IOU = 0.9  # Desired IoU between original and shifted box
MAX_ATTEMPTS = 100  # Maximum attempts to find a valid shifted box
NUM_OBJECTS_REQUIRED = 1  # Specify the exact number of objects you want in the images
OUTPUT_JSON_FILE = f"train_dpo_iou_{TARGET_IOU:.2f}_objects_{NUM_OBJECTS_REQUIRED}_llamafactory.json"
# --- -------------- ---

# Check if paths exist
if not os.path.isdir(IMAGE_DIR):
    print(f"ERROR: Image directory not found: {IMAGE_DIR}")
    exit()
if not os.path.isfile(ANNOTATION_FILE):
    print(f"ERROR: Annotation file not found: {ANNOTATION_FILE}")
    exit()

# Load COCO annotations
print(f"Loading annotations from: {ANNOTATION_FILE}")
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()
print(f"Found {len(image_ids)} images in the annotation file.")

# Create a mapping from category ID to category name
cat_ids = coco.getCatIds()
categories_map = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
print(f"Loaded {len(categories_map)} categories.")

# Count the number of valid annotations (with bbox and category_id) for each image
image_annotation_counts = defaultdict(int)
all_anns = coco.loadAnns(coco.getAnnIds())
for ann in all_anns:
    if 'bbox' in ann and 'category_id' in ann:
        image_annotation_counts[ann['image_id']] += 1

# Filter image_ids to include only those with the specified number of objects
filtered_image_ids = [
    img_id for img_id in image_ids if image_annotation_counts.get(img_id, 0) == NUM_OBJECTS_REQUIRED
]
print(f"Found {len(filtered_image_ids)} images with exactly {NUM_OBJECTS_REQUIRED} objects.")

def calculate_iou(bbox1, bbox2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1: [xmin, ymin, xmax, ymax]
        bbox2: [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection coordinates
    x_i = max(x1_1, x1_2)
    y_i = max(y1_1, y1_2)
    w_i = max(0, min(x2_1, x2_2) - x_i)
    h_i = max(0, min(y2_1, y2_2) - y_i)
    intersection_area = w_i * h_i

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def shift_bbox_with_iou(original_bbox, image_width, image_height, target_iou=0.9, max_attempts=100):
    """
    Shifts the bounding box at a random angle until the IoU with the original box is close to the target.

    Args:
        original_bbox: [xmin, ymin, xmax, ymax] of the original bounding box.
        image_width: Width of the image.
        image_height: Height of the image.
        target_iou: The desired IoU between the original and shifted box.
        max_attempts: Maximum attempts to find a valid shifted box.

    Returns:
        list or None: [xmin, ymin, xmax, ymax] of the shifted bounding box (floats), or None if not found.
    """
    ox1, oy1, ox2, oy2 = original_bbox
    original_width = ox2 - ox1
    original_height = oy2 - oy1

    for _ in range(max_attempts):
        # Generate a random angle in radians
        angle = random.uniform(0, 2 * math.pi)

        # Generate a random small shift distance
        max_shift = min(original_width, original_height) * 0.2  # Limit shift to a fraction of the size
        shift_magnitude = random.uniform(1, max_shift)

        # Calculate the shift in x and y directions
        shift_x = shift_magnitude * math.cos(angle)
        shift_y = shift_magnitude * math.sin(angle)

        # Apply the shift
        nx1 = max(0, min(image_width, ox1 + shift_x))
        ny1 = max(0, min(image_height, oy1 + shift_y))
        nx2 = max(0, min(image_width, ox2 + shift_x))
        ny2 = max(0, min(image_height, oy2 + shift_y))

        shifted_bbox = [nx1, ny1, nx2, ny2]

        # Calculate the IoU
        iou = calculate_iou(original_bbox, shifted_bbox)

        # Check if the IoU is close to the target
        if abs(iou - target_iou) < 0.05:  # Allow a small tolerance
            return shifted_bbox

    return None

def generate_prompt(categories):
    """Generates the prompt string based on the categories present."""
    categories_list = ", ".join(sorted(list(set(categories)))) # Sort for consistency
    return (
        f"Detect all {categories_list} in the image and return their locations in the form of coordinates. "
        "For each object category, provide a dictionary where keys are category names and values are lists of "
        "bounding boxes in the format [xmin, ymin, xmax, ymax]. "
        "Example format: {{'category1': [[x1,y1,x2,y2], [x1,y1,x2,y2]], 'category2': [[x1,y1,x2,y2]]}}. "
        "Only respond with the dictionary, no additional text."
    )

output_data = [] # Initialize an empty list to store the DPO entries
processed_images = 0
skipped_images_file_missing = 0
skipped_images_no_anns = 0
skipped_negative_generation = 0
image_ids_with_no_annotations = []

for img_id in tqdm(filtered_image_ids, desc=f"Processing COCO train images with {NUM_OBJECTS_REQUIRED} objects for DPO"):
    # Load image info
    try:
        image_info = coco.loadImgs(img_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image_width = image_info['width']
        image_height = image_info['height']
    except IndexError:
        print(f"Warning: Could not load image info for ID {img_id}. Skipping.")
        continue

    # Check if the image file actually exists
    if not os.path.exists(image_path):
        skipped_images_file_missing += 1
        continue

    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    chosen_bbox_dict = defaultdict(list)
    rejected_bbox_dict = defaultdict(list)
    present_category_names = set()

    if not annotations:
        skipped_images_no_anns += 1
        image_ids_with_no_annotations.append(img_id)
        prompt_text = "Detect all objects in the image and return their locations in the form of coordinates. For each object category, provide an empty dictionary."
        chosen_response_text = "{}"
        rejected_response_text = "{}"
        dpo_entry = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{prompt_text}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response_text
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response_text
            }
        }
        output_data.append(dpo_entry)
        processed_images += 1
        continue

    valid_annotations = [ann for ann in annotations if 'bbox' in ann and 'category_id' in ann]
    if len(valid_annotations) != NUM_OBJECTS_REQUIRED:
        continue # Skip if the exact number of valid objects is not met

    for ann in valid_annotations:
        category_id = ann['category_id']
        if category_id in categories_map:
            category_name = categories_map[category_id]
            present_category_names.add(category_name)

            # COCO bbox format: [xmin, ymin, width, height]
            x, y, w, h = ann['bbox']
            original_bbox = [float(x), float(y), float(x + w), float(y + h)] # Ensure original are floats
            chosen_bbox_dict[category_name].append(original_bbox)

            # Generate rejected bbox with target IoU
            rejected_bbox = shift_bbox_with_iou(original_bbox, image_width, image_height, target_iou=TARGET_IOU, max_attempts=MAX_ATTEMPTS)
            if rejected_bbox:
                rejected_bbox_dict[category_name].append(rejected_bbox)
            else:
                skipped_negative_generation += 1
                # Removed the warning print statement as requested

    # If no valid categories/bboxes were found after processing annotations
    if not present_category_names or not chosen_bbox_dict:
        skipped_images_no_anns += 1
        image_ids_with_no_annotations.append(img_id)
        prompt_text = "Detect all objects in the image and return their locations in the form of coordinates. For each object category, provide an empty dictionary."
        chosen_response_text = "{}"
        rejected_response_text = "{}"
        dpo_entry = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{prompt_text}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response_text
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response_text
            }
        }
        output_data.append(dpo_entry)
        processed_images += 1
        continue

    # Generate the prompt
    prompt_text = generate_prompt(present_category_names)

    # Format the chosen and rejected response dictionaries as strings
    chosen_response_text = str(dict(chosen_bbox_dict))
    rejected_response_text = str(dict(rejected_bbox_dict))

    # Create the LLaMA Factory preference structure
    dpo_entry = {
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{prompt_text}"
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen_response_text
        },
        "rejected": {
            "from": "gpt",
            "value": rejected_response_text
        }
    }

    output_data.append(dpo_entry)
    processed_images += 1

# Write the entire list to a single JSON file
print(f"\nWriting {len(output_data)} entries to {OUTPUT_JSON_FILE}...")
with open(OUTPUT_JSON_FILE, 'w') as f:
    json.dump(output_data, f, indent=2) # Use json.dump to write the list

print("\n--- DPO Data Generation Summary ---")
print(f"Total images in annotation file: {len(image_ids)}")
print(f"Number of images processed (with exactly {NUM_OBJECTS_REQUIRED} objects): {processed_images}")
print(f"Skipped (image file not found): {skipped_images_file_missing}")
print(f"Skipped (no valid annotations): {skipped_images_no_anns}")
print(f"Failed to generate negative boxes with target IoU: {skipped_negative_generation}")
if image_ids_with_no_annotations:
    print(f"Image IDs with no valid annotations: {image_ids_with_no_annotations}")
print(f"Output JSON file generated: {OUTPUT_JSON_FILE}")
print("Done.")
