1. Download coco train2017, val2017, and annotations_trainval2017 from the coco website:
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

2. Create a huggingface style dataset from coco. The script is named create_coco_nobj_grpo.py. You need to specify the nobj variable in it. Currently it is set to 3 which means that it will create
   a subset of the dataset that has only 3 objects per image.

3. In src/r1-v/src/open_r1/coco_grpo.py, in the load_from_disk function, change the path to the one created from step2. Here we load the dataset we created in step2.

4. In the trainer, model = '.....', give the path of the model if you have downloaded it.

5. The iou reward function is called 'iou_reward'. You can modify it accordingly.

6. To run the code, first cd into src/r1-v, then run:
torchrun --nproc_per_node=4     --nnodes=1     --node_rank=0     --master_addr=127.0.0.1     --master_port=12345     src/open_r1/coco_grpo.py     --output_dir outputs/qwen2.5vl-3b-coco-grpo     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct     --dataset_name leonardPKU/clevr_cogen_a_train     --deepspeed local_scripts/zero3.json     --max_prompt_length 512     --max_completion_length 256     --per_device_train_batch_size 1     --gradient_accumulation_steps 2     --logging_steps 1     --bf16     --bf16_full_eval     --torch_dtype bfloat16     --report_to wandb     --gradient_checkpointing false     --attn_implementation flash_attention_2     --max_pixels 401408     --max_steps 100     --run_name Qwen2-VL-3B-GRPO-CLEVR     --save_steps 100     --save_only_model true     --num_generations 2 --reward_funcs iou

7. During installation, flash attention might not work. Two things to note here: 1) make sure CUDA is loaded. I have tried CUDA 12.4.0 and CUDA 12.6.0 and both work fine. version 2.7.2.post1 worked for me. The latest release didnt. Also, include the --no-build-isolation flag while installing.

8. While running the main code, you might encounter an error saying mismatch between datatypes, float16 and float32. For that you need to change the lines:
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)

   to:

    q_embed = apply_rotary_emb(q, cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k, cos.float(), sin.float()).type_as(k)

   in transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py

   You will have to check where your library is installed. usually it is something like python3.x/site-packages/..../.../../transformers. The error will give you the exact path.

9. To run with peft:
    torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    src/open_r1/coco_grpo.py \
    --output_dir outputs/qwen2.5vl-3b-coco-grpo-3obj \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --bf16_full_eval \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-3B-GRPO-CLEVR \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 2 \
    --reward_funcs iou \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_target_modules q_proj v_proj k_proj o_proj

