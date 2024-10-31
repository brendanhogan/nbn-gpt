"""
Script to evaluate a trained GPT model checkpoint on a validation dataset.

This script loads a saved model checkpoint and evaluates its performance on a validation set,
saving the validation metrics to a JSON file.

Example:
    $ python eval_pretrain.py --checkpoint path/to/checkpoint.pt --dataset fineweb
"""
import os
import json
import torch
import argparse

import utils
import models
import datasets
import tokenizers

def get_args() -> argparse.Namespace:
    """Parse command line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--dataset", type=str, default="fineweb", choices=["creepypasta", "fineweb"], 
                        help="Dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length for evaluation")
    parser.add_argument("--val_tokens", type=int, default=10485760, help="Number of tokens to use for validation")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=1994, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    # Get arguments
    args = get_args()
    
    # Setup device and distributed parameters (single GPU/CPU mode)
    distributed_params = utils.setup_distributed_training()
    
    # Set random seed
    utils.seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = tokenizers.get_tokenizer("gpt2")
    model = models.get_model('gpt2', vocab_size=tokenizer.vocab_size).to(distributed_params['device'])
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=distributed_params['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from step {checkpoint['step']}")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {str(e)}")
    
    # Initialize data loaders
    _, val_loader = datasets.get_data_loaders(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        sequence_length=args.context_length,
        split_ratio=0.9,
        process_rank=distributed_params['rank'],
        num_processes=distributed_params['world_size']
    )
    
    # Calculate number of validation steps
    number_of_validation_steps = args.val_tokens // (args.batch_size * args.context_length * distributed_params['world_size'])
    
    # Evaluate model
    val_loss = utils.evaluate_model(
        model=model,
        val_loader=val_loader,
        val_steps=number_of_validation_steps,
        device=distributed_params['device'],
        global_step=checkpoint['step'],
        distributed_params=distributed_params
    )
    
    # Evaluate on HellaSwag
    hw_acc = utils.evaluate_on_hellaswag(distributed_params, model)
    
    # Save results
    results = {
        'checkpoint_path': args.checkpoint,
        'checkpoint_step': checkpoint['step'],
        'validation_loss': val_loss.item(),
        'hellaswag_accuracy': hw_acc,
        'dataset': args.dataset,
        'validation_tokens': args.val_tokens,
        'batch_size': args.batch_size,
        'context_length': args.context_length
    }
    
    output_file = os.path.join(args.output_dir, f"eval_results_step_{checkpoint['step']}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"HellaSwag Accuracy: {hw_acc:.4f}")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
