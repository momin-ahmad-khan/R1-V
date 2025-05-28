import os
import PIL
import json
import ast
from peft import PeftModel
from pycocotools.coco import COCO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

class COCOEvaluator:
    def __init__(self, model_path, coco_dir, annotation_file, output_file):
        self.model_path = model_path
        #self.adapter_path = '/leonardo_scratch/fast/EUHPC_D18_005/momin/qwen-peft-v6/checkpoint-96'
        # self.adapter_path = '/leonardo/home/userexternal/mkhan001/models/qwen-peft-v25/checkpoint-435'
        self.coco_dir = coco_dir
        self.annotation_file = annotation_file
        self.output_file = output_file
        self.results = {}

        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # Count the number of annotations per image
        image_annotation_counts = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            image_annotation_counts[image_id] = image_annotation_counts.get(image_id, 0) + 1

        # Filter images that have exactly one annotation
        self.single_object_image_ids = {img_id for img_id, count in image_annotation_counts.items() if count == 7}
        print(len(self.single_object_image_ids))
        self.single_object_image_ids = set(self.single_object_image_ids)

        # Initialize model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2.5-VL-3B-Instruct',
            torch_dtype="auto",
            device_map="auto"
        )

        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        #print("Model vision tower:", getattr(self.model, 'vision_tower', 'No vision tower found'))
        #self.model = PeftModel.from_pretrained(self.model, "/leonardo_scratch/fast/EUHPC_D18_005/momin/qwen-test/checkpoint-821")
        self.processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
        #self.processor.tokenizer.model_max_length = self.model.config.max_length
        self.model.load_adapter('/ceph/hpc/data/d2024d11-005-users/momin/R1-V/src/r1-v/outputs/qwen2.5vl-3b-coco-grpo-7obj_fiomcrR/checkpoint-600')
        # self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()

    def generate_prompt(self, categories):
        categories_list = ", ".join(categories)
        return (
            f"Please output bbox coordinates and names of {categories_list}."
        )

    # def parse_output(self, output_text):
    #     try:
    #         # Extract the first dictionary-like structure from the output
    #         start_idx = output_text.find('{')
    #         end_idx = output_text.rfind('}') + 1
    #         dict_str = output_text[start_idx:end_idx]
    #         return ast.literal_eval(dict_str)
    #     except (ValueError, SyntaxError) as e:
    #         print(f"Error parsing output: {str(e)}")
    #         return None

    def parse_output(self, output_text):
        try:
            # Strip markdown code block if present
            if output_text.strip().startswith("```json"):
                output_text = output_text.strip().removeprefix("```json").removesuffix("```").strip()

            parsed = json.loads(output_text)
            if not isinstance(parsed, list):
                raise ValueError("Parsed output is not a list")

            grouped = {}
            for obj in parsed:
                label = obj.get("label")
                bbox = obj.get("bbox_2d")
                if label and isinstance(bbox, list) and len(bbox) == 4:
                    grouped.setdefault(label, []).append(bbox)

            return grouped
        except Exception as e:
            print(f"[WARN] Failed to parse model output: {e}")
            return None


    def process_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.coco_dir, image_info['file_name'])

        # Get categories present in the image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        category_ids = {ann['category_id'] for ann in annotations}
        categories = [self.coco.loadCats(cat_id)[0]['name'] for cat_id in category_ids]

        if not categories:
            return None

        try:
            # Prepare VLM input
            prompt = self.generate_prompt(categories)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt}
                ]
            }]

            # Process inputs
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            #print("vision processed")
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            print(output_text)
            parsed_results = self.parse_output(output_text)
            if parsed_results:
                results = {
                    "image_id": image_id,
                    "image_path": img_path,
                    "detections": parsed_results
                }
                self.save_results(image_id, results)
                return results

        except Exception as e:
            print(f"\nError processing {image_id}: {str(e)}")
            return None

    def save_results(self, image_id, results):
        # Update results and save incrementally
        self.results[image_id] = results
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=4)

    def evaluate(self):
        #self.image_ids = self.image_ids[:5000]
        #for image_id in tqdm(self.image_ids, desc="Processing COCO images"):
        for image_id in tqdm(self.single_object_image_ids, desc="Processing COCO images"):
            if image_id not in self.results:  # Skip already processed images
                self.process_image(image_id)

if __name__ == "__main__":
    # Configuration
    evaluator = COCOEvaluator(
        model_path="/ceph/hpc/data/d2024d11-005-users/momin/R1-V/src/r1-v/outputs/qwen2.5vl-3b-coco-grpo-3obj-new/checkpoint-1100/PR1-Qwen2.5-VL-3B-Detection",
        coco_dir="/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/val2017",
        annotation_file="/ceph/hpc/data/d2024d11-005-users/momin/datasets/coco/annotations/instances_val2017.json",
        # output_file="results_5obj_fiomcr/vlm_coco_results_base_5obj.json"
        output_file = "results_7obj_fiomcrR/vlm_coco_results_grpo_7obj_chkpt600.json"
    )

    # Start evaluation
    evaluator.evaluate()
    print(f"\nEvaluation complete. Results saved to {evaluator.output_file}")
