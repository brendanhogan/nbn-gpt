# 124 M model 
#torchrun --standalone --nproc_per_node=8 main_pretrain.py --dataset fineweb --use_wb_tracking True

# 1.5 B model
# torchrun --standalone --nproc_per_node=8 main_pretrain.py --dataset fineweb --use_wb_tracking True --total_batch_size 491520 --batch_size 12 --max_steps 15258 --learning_rate 0.0018 --warmdown_iters 4359 --model_name gpt2full
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=2 main_pretrain.py --dataset fineweb --total_batch_size 491520 --batch_size 12 --max_steps 15258 --learning_rate 0.0018 --warmdown_iters 4359 --model_name gpt2full
