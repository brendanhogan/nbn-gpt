# 124 M model 
# python main_pretrain.py --dataset fineweb --use_wb_tracking True
# 1.5B model 
python main_finetune.py --dataset creepypasta --total_batch_size 491520 --batch_size 12 --max_steps 50 --learning_rate 0.00018 --warmdown_iters 10 --model_name gpt2full  --base_model output/step_15256/checkpoint.pt
