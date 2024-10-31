# 124 M model 
# python main_pretrain.py --dataset fineweb --use_wb_tracking True
# 1.5B model 
# python main_pretrain.py --dataset fineweb --use_wb_tracking True --total_batch_size 491520 --batch_size 12 --max_steps 15258 --learning_rate 0.0018 --warmdown_iters 4359 --model_name gpt2full
python main_pretrain.py --dataset fineweb --total_batch_size 491520 --batch_size 12 --max_steps 15258 --learning_rate 0.0018 --warmdown_iters 4359 --model_name gpt2full
