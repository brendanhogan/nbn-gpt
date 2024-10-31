import os
import math
import json
import time
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm 
from datetime import datetime
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

# For pdf plotting
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Torch distributed stuff 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Modules from this codebasse
import datasets
import tokenizers
import data.hellaswag
import data.baselines

###################
### TRAIN UTILS ##
##################

def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """
    From: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L9
    Compute the most likely row by evaluating autoregressive loss at all positions.
    
    This function calculates token-level cross entropy losses, masks them according to 
    completion regions, and returns the index of the row with lowest average loss.

    Args:
        tokens (torch.Tensor): Input token ids of shape [batch_size, seq_len]
        mask (torch.Tensor): Binary mask of shape [batch_size, seq_len] indicating completion regions
        logits (torch.Tensor): Model logits of shape [batch_size, seq_len, vocab_size]

    Returns:
        int: Index of the row with lowest average masked loss
    """
    # Shift and flatten logits/tokens for cross entropy calculation
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    # Calculate token-level losses
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # Mask losses to only consider completion regions
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask

    # Calculate average loss per row
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # Return index of row with minimum average loss
    return avg_loss.argmin().item()

def evaluate_on_hellaswag(distributed_params: Dict[str, Any], model: torch.nn.Module) -> Optional[float]:
    """
    Evaluates model performance on the HellaSwag benchmark dataset in a distributed setting.
    
    This function:
    1. Processes validation examples across distributed processes
    2. Makes predictions using the model in a memory-efficient way
    3. Aggregates accuracy metrics across all processes
    4. Returns accuracy score on master process
    
    Args:
        distributed_params: Dictionary containing distributed training parameters including:
            - world_size: Total number of processes
            - rank: Current process rank
            - device: Device to run evaluation on
            - device_type: Type of device (cuda/cpu)
            - running_distributed: Whether running in distributed mode
            - master_process: Whether this is the master process
        model: The PyTorch model to evaluate
            
    Returns:
        float: Accuracy score on HellaSwag validation set (only on master process)
        None: On non-master processes
    """
    # Initialize metrics
    num_correct_norm = 0
    num_total = 0
    
    # Iterate through validation set
    for i, example in tqdm(enumerate(data.hellaswag.iterate_examples("val")), total=10042, desc="Evaluating HellaSwag"):
        # Process examples according to process rank
        if distributed_params['running_distributed']:
            if i % distributed_params['world_size'] != distributed_params['rank']:
                continue
                
        # Get tokens and labels for example
        _, tokens, mask, label = data.hellaswag.render_example(example)
        tokens = tokens.to(distributed_params['device'])
        mask = mask.to(distributed_params['device'])
        
        # Get model predictions
        with torch.no_grad():
            with torch.autocast(device_type=distributed_params['device_type'], dtype=torch.bfloat16):
                logits, _ = model(tokens)
            
            pred_norm = get_most_likely_row(tokens, mask, logits)
            
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # Aggregate statistics across processes
    if distributed_params['running_distributed']:
        num_total = torch.tensor(num_total, dtype=torch.long, device=distributed_params['device'])
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=distributed_params['device'])
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()

    # Calculate and return accuracy (master process only)
    acc_norm = num_correct_norm / num_total
    if distributed_params['master_process']:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        return acc_norm
    return None
  
def evaluate_model(model: torch.nn.Module, val_loader: datasets.AbstractDataLoader, val_steps: int, device: torch.device, global_step: int, distributed_params: Dict[str, any]) -> float:
    """
    Evaluate a PyTorch model on validation data.

    This function evaluates the model by computing the average loss over a specified
    number of validation steps. The model is put in evaluation mode and no gradients
    are computed during evaluation.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        val_loader (datasets.AbstractDataLoader): DataLoader containing validation data.
        device (torch.device): Device to run evaluation on (CPU/GPU).
        val_steps (int, optional): Number of validation steps to average over. Defaults to 20.
        global_step (int): Current global training step for logging.
        distributed_params (int): Distributed info.

    Returns:
        float: Average validation loss over the specified number of steps.
    """
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    
    # with torch.no_grad():
    for _ in tqdm(range(val_steps), desc="Evaluating", disable=not distributed_params['master_process']):
        x, y = val_loader.get_batch()
        with torch.autocast(device_type=distributed_params['device_type'], dtype=torch.bfloat16):
            _, loss = model(x, y, return_logits=False)
            val_loss_accum += loss.detach()
            del loss

    # If distributed average over all GPUs
    if distributed_params["running_distributed"]:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)    

    # Average over steps    
    val_loss_accum = val_loss_accum / val_steps
            
    return val_loss_accum

def generate_samples(args, step: int, model: torch.nn.Module, tokenizer: tokenizers.AbstractTokenizer, device: torch.device, distributed_params: Dict[str, any], num_samples: int = 4, max_length: int = 32, prompt: str = "It was a dark and stormy night ") -> List[str]:
    """
    Generate text samples from a trained language model using top-k sampling.

    Args:
        step (int): The current training step 
        model (torch.nn.Module): The trained language model to generate samples from.
        tokenizer (tokenizers.AbstractTokenizer): The tokenizer used to encode/decode text.
        device (torch.device): The device (CPU/GPU) to run generation on.
        num_samples (int, optional): Number of text samples to generate. Defaults to 4.
        max_length (int, optional): Maximum length of generated sequences in tokens. Defaults to 32.
        distributed_params Dict: distributed paramters 
        prompt (str, optional): Text prompt to condition the generation. 
            Defaults to "It was a dark and stormy night,".

    Returns:
        list[str]: List of generated text samples, each starting with the given prompt.
    """
    # model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1).cuda()

    # TODO: This should probably be an argument - if you want deterministic generations or not
    # sample_rng = torch.Generator(device=distributed_params['device']).manual_seed(7111994)
    
    with torch.no_grad():
        while tokens.size(1) < max_length:
            with torch.autocast(device_type=distributed_params['device_type'], dtype=torch.bfloat16):
                logits, _ = model(tokens)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            next_token = torch.gather(topk_indices, -1, torch.multinomial(topk_probs, 1))
            tokens = torch.cat((tokens, next_token), dim=1)
            del logits 

    generated_samples = []
    for i in range(num_samples):
        decoded = tokenizer.decode(tokens[i, :max_length].tolist())
        generated_samples.append(decoded)
        
        # Log each sample to W&B with step number
        if args.use_wb_tracking:
            wandb.log({
                f"samples/text_{i+1}": wandb.Html(decoded.replace("\n", "<br>")),
                "samples/step": step,
                "global_step": step
            }, step=step)
    
    del tokens 
    # del sample_rng
    
    return generated_samples

def train_step(model: torch.nn.Module, train_loader: datasets.AbstractDataLoader, device: torch.device, grad_accum_steps: int, distributed_params:Dict[str, any]) -> torch.Tensor:
    """
    Perform a single training step with gradient accumulation.

    This function executes one complete training step, which consists of multiple micro-steps
    for gradient accumulation. It handles moving data to the appropriate device, computing loss,
    and accumulating gradients before returning the total loss.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        train_loader (datasets.AbstractDataLoader): DataLoader providing training batches.
        device (torch.device): The device (CPU/GPU) to perform computations on.
        grad_accum_steps (int): Number of gradient accumulation steps before updating weights.
        distributed_params Dict: Parameters for distributed training 

    Returns:
        torch.Tensor: The accumulated loss value for this training step.
    """

    model.train()
    loss_accum = 0.0
    
    for grad_accum_step in range(1, grad_accum_steps+1):
        x, y = train_loader.get_batch()
        with torch.autocast(device_type=distributed_params['device_type'], dtype=torch.bfloat16):
            _, loss = model(x, y, return_logits=False)
            loss_accum += loss.detach()

        # If disitrubted, and we are not in the last step, do not want to sync grads 
        if distributed_params["running_distributed"] and grad_accum_step < grad_accum_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    # average out gradients over accumulation steps 
    for p in model.parameters():
        p.grad /= grad_accum_steps

    # # Average out loss across alll GPUs
    if distributed_params["running_distributed"]:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)    
    loss_accum = loss_accum / grad_accum_steps

    return loss_accum

def resume_from_checkpoint(args: argparse.Namespace, model: torch.nn.Module, optimizers: torch.optim.Optimizer, device: torch.device, distributed_params: Dict[str, any]) -> int:
    """
    Resumes model training from the latest checkpoint if resume_training is enabled.

    This function searches for the most recent checkpoint in the output directory,
    loads the model and optimizer states, and returns the next training step number.
    If no checkpoint is found or loading fails, appropriate exceptions are raised.

    Args:
        args (argparse.Namespace): Arguments containing resume_training flag and output_dir
        model (torch.nn.Module): The model to load the saved state into
        optimizer (torch.optim.Optimizer): The optimizer to load the saved state into
        device (torch.device): The device to load the checkpoint onto
        distributed_params: Dict[str, any]: Distributed parameters 

    Returns:
        int: The next training step number (0 if not resuming)

    Raises:
        ValueError: If no checkpoint is found, the checkpoint file is missing,
                   or if loading the checkpoint fails
    """

    # TODO: This loading works - but also need to be sure to return right starting step 
    # and needs to step the learning rate schedulers, and training dataset appropraite 
    # number of times. 

    if args.resume_training:
        # Find the highest step directory with a checkpoint
        step_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("step_")]
        if not step_dirs:
            raise ValueError("No checkpoint found for resuming training.")
        
        latest_step_dir = max(step_dirs, key=lambda x: int(x.split("_")[1]))
        checkpoint_path = os.path.join(args.output_dir, latest_step_dir, "checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found in {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=distributed_params['device'])
            distributed_params['raw_model'].load_state_dict(checkpoint['model_state_dict'])
            optimizers[0].load_state_dict(checkpoint['optimizer1_state_dict'])
            optimizers[1].load_state_dict(checkpoint['optimizer2_state_dict'])
            start_step = checkpoint['step'] + 1
            
            # Log checkpoint restoration
            if args.use_wb_tracking:
                wandb.log({
                    "checkpoint/restored_from": checkpoint_path,
                    "checkpoint/restored_step": start_step - 1,
                    "checkpoint/val_loss": checkpoint['val_loss'],
                    "global_step": start_step - 1
                }, step=start_step-1)
            
            print(f"Resuming training from step {start_step}")
            return start_step
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {str(e)}")
    return 0

def save_checkpoint(model: torch.nn.Module, optimizers: torch.optim.Optimizer, val_loader: datasets.AbstractDataLoader, tokenizer: tokenizers.AbstractTokenizer, device: torch.device, step: int, number_of_validation_steps: int, args: argparse.Namespace, distributed_params: Dict[str, any]) -> None:
    """
    Saves a training checkpoint and evaluation metrics at the current training step.

    This function performs the following:
    1. Creates a step-specific directory and saves validation loss
    2. Optionally generates and saves model output samples
    3. Periodically saves model and optimizer states
    4. Saves training configuration

    Args:
        model: The neural network model being trained
        optimizer: The optimizer used for training
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer for text generation
        device: Device (CPU/GPU) to run computations on
        step: Current training step number
        number_of_validation_steps: Number of validation steps to evaluate
        args: Namespace containing training arguments and configuration
        distributed_params: Distributed info dict 

    Returns:
        None
    """
    # TODO: Make sure everything is automatically save on last step
    # Create a directory for this step
    step_dir = os.path.join(args.output_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    # Evaluate and save validation loss
    print("evaluating val loss")
    val_loss = evaluate_model(model, val_loader, number_of_validation_steps, device, step, distributed_params)

    # Evaluate on hellaswag
    ## TODO: SKIPPING HELLASWAG FOR NOW -- doesnt work with compile 
    # hw_acc = evaluate_on_hellaswag(distributed_params, model)
    hw_acc = .1


    # Only do this logging stuff if is the main process
    if distributed_params['master_process']:
        # Calculate total elapsed time since training started
        elapsed_time = time.time() - distributed_params['training_start_time']
        
        # Log validation metrics to W&B
        # Get baseline values for comparison
        gpt2_loss_baseline = data.baselines.loss_baseline['124M']
        gpt2_hellaswag_same_size_baseline = data.baselines.hella2_baseline['124M']
        gpt3_hellaswag_same_size_baseline = data.baselines.hella3_baseline['124M']
        
        gpt2_hellaswag_fullsize_baseline = data.baselines.hella2_baseline['1558M']
        gpt3_hellaswag_fullsize_baseline = data.baselines.hella3_baseline['1558M']
        
        if args.use_wb_tracking:
            wandb.log({
                # Current model metrics
                "hellaswag/acc": hw_acc,
                "val/loss": val_loss.item(),
                "time/elapsed_seconds": elapsed_time,
                
                # Loss baselines
                "baselines/gpt2_124M_loss": gpt2_loss_baseline,
                
                # HellaSwag baselines - same model size (124M)
                "baselines/gpt2_124M_hellaswag": gpt2_hellaswag_same_size_baseline,
                "baselines/gpt3_124M_hellaswag": gpt3_hellaswag_same_size_baseline,
                
                # HellaSwag baselines - full size models (1.5B)
                "baselines/gpt2_1.5B_hellaswag": gpt2_hellaswag_fullsize_baseline,
                "baselines/gpt3_1.5B_hellaswag": gpt3_hellaswag_fullsize_baseline,
                
                # Step tracking
                "val/step": step,
                "global_step": step
            }, step=step)
        with open(os.path.join(step_dir, "val_loss.json"), "w") as f:
            json.dump({
                "validation_loss": val_loss.item(), 
                "hellaswag_acc": hw_acc,
                "elapsed_time_seconds": elapsed_time
            }, f)

        print(f"Step {step}: Validation Loss: {val_loss:.4f} HellaSwag Acc: {hw_acc:.4f} Elapsed Time: {elapsed_time:.2f}s")

        # TODO: SKIP GENERATION FOR NOW AS WELL - same compile issue        
        # if args.generate_samples:
        #     # Generate and save samples
        #     samples = generate_samples(args, step, model, tokenizer, device, distributed_params)
            
        #     # Save generated samples as JSON
        #     with open(os.path.join(step_dir, "generated_samples.json"), "w") as f:
        #         json.dump({
        #             "generated_samples": samples,
        #             "elapsed_time_seconds": elapsed_time
        #         }, f)
            
        #     # Generate PDF for readability
        #     pdf_filename = os.path.join(step_dir, "generated_samples.pdf")
        #     create_pdf_from_samples(samples, pdf_filename)
            

        # Save model and optimizer state
        if step % args.model_save_interval == 0:
            checkpoint_path = os.path.join(step_dir, "checkpoint.pt")
            torch.save({
                'step': step,
                'model_state_dict': distributed_params['raw_model'].state_dict(),
                'optimizer1_state_dict': optimizers[0].state_dict(),
                'optimizer2_state_dict': optimizers[1].state_dict(),
                'val_loss': val_loss,
                'elapsed_time_seconds': elapsed_time
            }, checkpoint_path)
            
            # Log checkpoint as artifact
            if args.use_wb_tracking:
                wandb.log({
                    "checkpoint/saved_path": checkpoint_path,
                    "checkpoint/step": step,
                    "checkpoint/val_loss": val_loss.item(),
                    "checkpoint/elapsed_time_seconds": elapsed_time,
                    "global_step": step
                }, step=step)

        # Save training configuration
        config_path = os.path.join(step_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f)
        
        # Log config as artifact
        if args.use_wb_tracking:
            wandb.log({
                "config/path": config_path,
                "global_step": step
            }, step=step)

def train_model(
    model: torch.nn.Module,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[any],
    train_loader: datasets.AbstractDataLoader,
    val_loader: datasets.AbstractDataLoader,
    device: torch.device,
    args: argparse.Namespace,
    tokenizer: tokenizers.AbstractTokenizer,
    grad_accum_steps: int,
    number_of_validation_steps: int, 
    distributed_params: Dict[str, any]
) -> None:
    """
    Main training loop for model training.

    This function handles the complete training process including:
    - Resuming from checkpoints if specified
    - Executing training steps with gradient accumulation
    - Periodic model evaluation and checkpointing
    - Learning rate scheduling
    - Progress logging and metrics tracking

    Args:
        model (torch.nn.Module): The neural network model to train
        optimizers List(torch.optim.Optimizer): Both optimizers for model parameter updates
        schedulers List(torch.optim.lr_scheduler): Both schedulers for model parameter lr
        train_loader (datasets.AbstractDataLoader): DataLoader for training data
        val_loader (datasets.AbstractDataLoader): DataLoader for validation data
        device (torch.device): Device (CPU/GPU) to run computations on
        args (argparse.Namespace): Training configuration and hyperparameters
        tokenizer (tokenizers.AbstractTokenizer): Tokenizer for text processing
        grad_accum_steps (int): Number of gradient accumulation steps
        number_of_validation_steps (int): Number of validation steps to evaluate
        distributed_params: Dict[str, any]: Distributed information

    Returns:
        None
    """
    # Initialize the starting step
    if args.resume_training:
        start_step = resume_from_checkpoint(args, model, optimizers, device, distributed_params)

    # Record training start time
    if distributed_params['master_process']:
        torch.cuda.synchronize()
        distributed_params['training_start_time'] = time.time()

    # Initialize training loader 
    train_loader.reset()

    # Step through iteration steps 
    for step in range(args.max_steps):

        # Initialize training metrics dictionary
        training_dict = {}
        
        # Evaluate model and save checkpoint at specified intervals
        if step % args.eval_interval == 0 or step == args.max_steps - 1 and distributed_params['master_process']:
            save_checkpoint(model, optimizers, val_loader, tokenizer, device, step, number_of_validation_steps, args, distributed_params)

        # Track iteration timing
        if distributed_params["master_process"]:
            training_dict['start_time'] = time.time()

        # Execute training step
        loss = train_step(model, train_loader, device, grad_accum_steps, distributed_params)
        training_dict['train_loss'] = loss.item()

        # Optimization and scheduler step
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        
        # Calculate timing and throughput metrics
        if distributed_params["master_process"]:
            torch.cuda.synchronize()

            training_dict['end_time'] = time.time()
            dt = training_dict['end_time'] - training_dict['start_time']
            elapsed_time = training_dict['end_time'] - distributed_params['training_start_time']

            tokens_processed = train_loader.batch_size * train_loader.sequence_length * grad_accum_steps * distributed_params['world_size']
            tokens_per_sec = tokens_processed / dt
            total_tokens = step * tokens_processed

            # Record metrics
            training_dict['time'] = f"{dt*1000:.2f}ms"
            training_dict['elapsed_time_seconds'] = elapsed_time
            training_dict['tokens_processed'] = tokens_processed
            training_dict['tokens_per_sec'] = tokens_per_sec
            training_dict['total_tokens_trained'] = total_tokens
            training_dict['step'] = step
            
            # Log all metrics to W&B
            if args.use_wb_tracking:
                wandb.log({
                    "train/loss": loss.item(),
                    "perf/step_time_ms": dt * 1000,
                    "perf/tokens_per_second": tokens_per_sec,
                    "perf/tokens_processed": tokens_processed,
                    "perf/total_tokens_trained": total_tokens,
                    "time/elapsed_seconds": elapsed_time,
                    "train/step": step,
                    "global_step": step
                }, step=step)
            
            # Log progress
            # Calculate remaining time based on current step and max steps
            steps_remaining = args.max_steps - step
            time_per_step = dt  # Current step time in seconds
            estimated_remaining_seconds = steps_remaining * time_per_step
            estimated_remaining_hours = estimated_remaining_seconds / 3600
            
            remaining_hours = int(estimated_remaining_hours)
            remaining_minutes = int((estimated_remaining_hours - remaining_hours) * 60)
            print(f"Step {step:5d}/{args.max_steps} | Loss: {loss.item():.6f} | Time: {dt*1000:.2f}ms | Elapsed: {elapsed_time/60:.2f}min | Remaining: {remaining_hours}h {remaining_minutes}m | Tokens/sec: {tokens_per_sec:.2f} | Total Tokens: {total_tokens:,}")
            # Save training metrics
            step_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            training_log_path = os.path.join(step_dir, "training_log.json")
            with open(training_log_path, "w") as f:
                json.dump(training_dict, f, indent=2)

            # Log training 
            if args.use_wb_tracking:
                wandb.log({
                    "logs/training_log_path": training_log_path,
                    "global_step": step
                }, step=step)



####################
## GENERIC UTILS ##
###################

def seed_everything(seed: int):
    """
    Set the seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings to ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def calculate_model_size(args: argparse.Namespace, model: torch.nn.Module) -> str:
    """
    Calculate and print the number of parameters in the model.

    Args:
        args (argparse.Namespace): Arguments containing output directory and W&B tracking flag
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        str: A string representation of the model size.
    """
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    size_str = f"{num_params:.2f}M parameters"
    print(size_str)
    
    # Save model size to file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "model_size.txt"), "w") as f:
        f.write(size_str)
    

    
    return size_str

def save_metadata(metadata: dict, output_dir: str, args) -> str:
    """
    Save metadata to a JSON file in the specified output directory.

    Args:
        metadata (dict): A dictionary containing metadata to be saved.
        output_dir (str): The directory where the metadata file will be saved.

    Returns:
        str: The path to the saved metadata file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Add current datetime to metadata
    metadata["datetime"] = datetime.now().isoformat()

    # Save metadata to JSON file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Log metadata to W&B
    if args.use_wb_tracking:
        wandb.config.update(metadata)
        wandb.log({
            "metadata/path": metadata_file,
            "global_step": 0
        }, step=0)

    print(f"Metadata saved to {metadata_file}")
    return metadata_file

def optimize_model(model: torch.nn.Module, distributed_params) -> torch.nn.Module:
    """
    Optimize a PyTorch model for faster training using various techniques.

    Args:
        model (torch.nn.Module): The PyTorch model to optimize.

    Returns:
        torch.nn.Module: The optimized PyTorch model.
    """

    # Enable PyTorch 2.0 compilation if available
    model = torch.compile(model)
    # Run a few batches through the model to trigger compilation
    model.train()
    for i in range(10):
        x = torch.randint(0, 50304, (12, 1024), device=distributed_params['device']) # Match batch size and seq len
        y = torch.randint(0, 50304, (12, 1024), device=distributed_params['device']) # Match batch size and seq len
        with torch.autocast(device_type=distributed_params['device_type'], dtype=torch.bfloat16):
            logits, loss = model(x,y)
            del logits 
            del loss 
        
    
    return model

def get_lr(it: int, args: argparse.Namespace) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_steps:
        return args.max_lr * (it + 1) / args.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.max_steps:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return args.min_lr + coeff * (args.max_lr - args.min_lr)

def create_pdf_from_samples(samples: List[str], pdf_filename: str) -> None:
    """
    Create a PDF file containing the generated samples.

    Args:
        samples (list): List of generated text samples.
        pdf_filename (str): Name of the output PDF file.

    Returns:
        None
    """

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for i, sample in enumerate(samples, 1):
        story.append(Paragraph(f"Sample {i}:", styles['Heading2']))
        story.append(Paragraph(sample, styles['BodyText']))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF created successfully: {pdf_filename}")

def setup_distributed_training() -> Dict[str, any]:
    """
    Sets up distributed training configuration and device settings.
    
    This function:
    1. Detects if running in distributed mode
    2. Initializes the distributed process group if needed
    3. Sets up device and rank information
    4. Returns all distributed training parameters in a dictionary
    
    Returns:
        Dict[str, any]: Dictionary containing distributed training parameters
    """
    # Check if running distributed
    running_distributed = int(os.environ.get('RANK', -1)) != -1
    
    if running_distributed:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # Local rank of GPU 
        ddp_world_size = int(os.environ['WORLD_SIZE']) # Number of GPUs
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # If we are first GPU -- use this one to do all logging
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print(f"using device: {device}")

    # Collect distributed training parameters into a single dictionary
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return {
        'running_distributed': running_distributed,
        'rank': ddp_rank,
        'local_rank': ddp_local_rank, 
        'world_size': ddp_world_size,
        'device': device,
        'master_process': master_process, 
        'device_type': device_type
    }

