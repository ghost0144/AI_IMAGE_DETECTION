#dinov3+vit16b+lora+mlp

CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 train.py
#train on SD1.4 and test on AIGIBench