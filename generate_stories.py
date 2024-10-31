import os
import torch
import models
import tokenizers
import utils
from typing import List
import argparse
from tqdm import tqdm

def generate_stories(model_type: str, output_dir: str, checkpoint_path: str, num_stories: int = 4, max_tokens: int = 300, batch_size: int = 4) -> List[str]:
    """
    Generate stories using a pretrained GPT model.

    Args:
        model_type (str): Type of model to use (e.g. "gpt2small", "gpt2full")
        output_dir (str): Directory to save generated stories
        checkpoint_path (str): Path to model checkpoint
        num_stories (int): Total number of stories to generate
        max_tokens (int): Maximum length of each story in tokens
        batch_size (int): Number of stories to generate in parallel

    Returns:
        List[str]: List of generated stories
    """
    # Initialize model and load checkpoint
    tokenizer = tokenizers.get_tokenizer("gpt2")
    model = models.get_model(model_name=model_type, vocab_size=tokenizer.vocab_size)
    
    checkpoint = torch.load(checkpoint_path)
    fixed_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(fixed_state_dict)
    model = model.cuda()
    model.eval()

    # Set up distributed params for generation
    distributed_params = {
        'running_distributed': False,
        'rank': 0,
        'local_rank': 0,
        'world_size': 1,
        'device': 'cuda',
        'master_process': True,
        'device_type': 'cuda'
    }

    # Generate stories in batches
    all_stories = []
    for i in tqdm(range(0, num_stories, batch_size), desc="Generating stories"):
        curr_batch_size = min(batch_size, num_stories - i)
        # Create minimal args object with required fields
        args = argparse.Namespace()
        args.use_wb_tracking = False
        stories = utils.generate_samples(
            args=args,
            step=0,     # Not needed for generation  
            model=model,
            tokenizer=tokenizer,
            device='cuda',
            distributed_params=distributed_params,
            num_samples=curr_batch_size,
            max_length=max_tokens,
            prompt="It was a dark and stormy night,"
        )
        all_stories.extend(stories)

    # Save stories to output directory
    os.makedirs(output_dir, exist_ok=True)
    for i, story in enumerate(all_stories):
        with open(os.path.join(output_dir, f"story_{i+1}.txt"), "w") as f:
            f.write(story)

    return all_stories

def main():
    parser = argparse.ArgumentParser(description='Generate stories using a trained GPT model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_stories',
                       help='Directory to save generated stories')
    parser.add_argument('--model_type', type=str, default='gpt2full',
                       help='Model architecture to use')
    parser.add_argument('--num_stories', type=int, default=10,
                       help='Number of stories to generate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')
    parser.add_argument('--max_tokens', type=int, default=300,
                       help='Maximum number of tokens per story')
    
    args = parser.parse_args()
    
    stories = generate_stories(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        num_stories=args.num_stories,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens
    )
    
    print(f"\nGenerated {len(stories)} stories in {args.output_dir}")

if __name__ == '__main__':
    main()

