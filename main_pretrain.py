"""
GPT-like Model Pretraining Script

This script implements the main training pipeline for pretraining a GPT-like language model
on text datasets. It handles:

- Distributed training setup across multiple GPUs using PyTorch DDP
- Model initialization and configuration
- Data loading and preprocessing
- Training loop with gradient accumulation
- Checkpointing and model evaluation
- Experiment tracking with Weights & Biases
- Reproducible training with seed setting

The training process includes:
- Learning rate scheduling with warmup
- Gradient clipping
- Mixed precision training
- Periodic model evaluation and sample generation
- Checkpoint saving and resumption

Example:
    $ python main_pretrain.py --dataset creepypasta --batch_size 64 --max_steps 20000

"""
import os
import json
import torch
import wandb
import argparse
from datetime import datetime

# Torch distributed stuff 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

import muon
import utils
import models 
import datasets 
import tokenizers

def get_args() -> argparse.Namespace:
    """
    Parse and return command line arguments for model training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing training configuration
    """
    parser = argparse.ArgumentParser(description="Pretrain a GPT-like model on  dataset")
    parser.add_argument("--model_name", type=str, default="gpt2small", choices=["gpt2small", "gpt2full"], help="Model architecture to use")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use for text encoding")
    parser.add_argument("--dataset", type=str, default="fineweb", choices=["creepypasta", "fineweb"], help="Dataset to use for training")
   
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (number input per gpu)")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="Total batch size in number of tokens for gradient propagation")
    parser.add_argument("--max_steps", type=int, default=5100, help="Maximum number of training steps")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length for the model")

    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--warmdown_iters", type=int, default=1450, help="Number of iterations for learning rate warmdown")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient for optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.0036, help="Learning rate for optimizer")
   
    parser.add_argument("--eval_interval", type=int, default=500, help="Number of steps between evaluations")
    parser.add_argument("--model_save_interval", type=int, default=1000, help="Number of steps between model checkpoints")
    parser.add_argument("--val_tokens", type=int, default=10420224, help="Number of tokens to use for validation")

    parser.add_argument("--compile_model", type=bool, default=True, help="Whether to compile model using torch.compile")
    parser.add_argument("--seed", type=int, default=1994, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--generate_samples", type=bool, default=True, help="Whether to generate samples during evaluation")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--use_wb_tracking", type=bool, default=False, help="Whether to use Weights & Biases for experiment tracking")
    args = parser.parse_args()
    return args 

@record # Can check distributed errors if need be 
def main() -> None:
    """
    Main training function that orchestrates the complete training pipeline.
    
    This function:
    1. Sets up distributed training if running on multiple GPUs
    2. Initializes model, optimizer, and data loaders
    3. Configures training parameters and logging
    4. Executes the main training loop
    """
    # 1. get args 
    args = get_args() 
    
    # 2. Setup disributed parameters (can handle single GPU training as well)    
    distributed_params = utils.setup_distributed_training()

    # 3. Initialize wandb if tracking is enabled
    if args.use_wb_tracking and distributed_params['master_process']:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.dataset}_{args.tokenizer}_{current_time}"
        wandb.init(
            project="gpt-training",
            name=experiment_name,
            config=vars(args),
            dir=args.output_dir
        )
        wandb.log({
            "distributed/is_distributed": distributed_params['running_distributed'],
            "distributed/world_size": distributed_params['world_size'],
        }, step=0)

    # 4. Save metadata for experiment
    if distributed_params['master_process']:
        metadata = vars(args)
        metadata_file = utils.save_metadata(metadata, args.output_dir, args)
        with open(os.path.join(args.output_dir, "distributed_params.json"), "w") as f:
            json.dump(distributed_params, f, indent=4)

    # 5. Set random seed for reproducibility
    utils.seed_everything(args.seed)

    # 6. Initialize tokenizer and model 
    tokenizer = tokenizers.get_tokenizer("gpt2")
    model = models.get_model(model_name=args.model_name, vocab_size=tokenizer.vocab_size)
    model = model.cuda()
    model_size = utils.calculate_model_size(args, model)
    
    # Use tensorcores 
    torch.set_float32_matmul_precision('high')
    if args.compile_model:
        model = torch.compile(model)

    # 7. Wrap in distrubted if enabled 
    if distributed_params['running_distributed']:
        model = DDP(model, device_ids=[distributed_params['local_rank']])
    raw_model = model.module if distributed_params['running_distributed'] else model
    distributed_params['raw_model'] = raw_model

    # 8. Initialize data loaders
    train_loader, val_loader = datasets.get_data_loaders(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        sequence_length=distributed_params['raw_model'].context_length,
        split_ratio=0.9,  # 90% for training, 10% for validation
        process_rank=distributed_params['rank'],
        num_processes=distributed_params['world_size']
    )


    # 9. Make sure grad accumulation and validation numbers are good, and set them up 
    assert args.total_batch_size % (args.batch_size * args.context_length * distributed_params['world_size']) == 0, "Make sure batch size is divisible to some number of small batches"
    grad_accum_steps = args.total_batch_size // ( args.batch_size * args.context_length * distributed_params['world_size'])
    number_of_validation_steps = args.val_tokens // (args.batch_size * args.context_length * distributed_params['world_size'])

    # 10. Setup optimizers
    optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
    optimizer2 = muon.Muon(raw_model.transformer_blocks.transformers.parameters(), lr=0.1*args.learning_rate, momentum=0.95, distributed_params=distributed_params)
    
    optimizer_list = [optimizer1, optimizer2]
    schedulers = muon.get_optim_schedulers(optimizer_list, args)
    
    # 11. Do training
    utils.train_model(model, optimizer_list, schedulers, train_loader, val_loader, distributed_params['device'], args, tokenizer, grad_accum_steps, number_of_validation_steps, distributed_params)


    # 12. Destory the processes if distributed 
    if distributed_params['running_distributed']:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

    





#
