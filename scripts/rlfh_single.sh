# 124 M model 
# python main_pretrain.py --dataset fineweb --use_wb_tracking True
# 1.5B model 
python main_finetune.py --dataset rlhf --total_batch_size 2048 --batch_size 2 --max_steps 3 --learning_rate 0.00018 --warmdown_iters 0 --model_name gpt2full  --base_model fine_tune_output/step_50/model.pt --output_dir rlfh_out
