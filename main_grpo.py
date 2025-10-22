"""

"""
import os
import json
import torch
import random
import argparse
from tqdm import trange, tqdm
from openai import OpenAI
from pydantic import BaseModel
import torch.nn.functional as F
from typing import Dict, List, Optional

# Own modules
import models
import tokenizers


# Setump prompts 
prepend_prompt = "I will tell you a scary four sentence story. Here is the story: "
SCARY_STORY_STARTERS = [
    "It was a dark and stormy night, and the old house at the end of Maple Street hadn't had a visitor in thirty years.",
    "The last thing I remember before waking up in the coffin was the sound of my own heartbeat slowing down.",
    "My daughter won't stop drawing the same figure in every picture—a tall man with no face standing just outside our window.",
    "The museum curator's face went pale when I showed her the photograph. 'That statue,' she whispered, 'has been in a different position every morning for the past week.'",
    "I inherited my grandmother's mirror, but my reflection always appears three seconds after I move.",
    "The town had a rule: never answer the door after midnight, no matter who—or what—is knocking.",
    "When the power went out, I used my phone's flashlight to navigate the basement, but in every photo I took, there was someone standing behind me.",
    "The antique music box only plays at 3:33 AM, and each night, the melody gets closer to my bedroom door.",
    "After my twin brother died, I started receiving texts from his number—each one counting down from 100.",
    "The search party found me after six days in the woods, but I have no memory of those days, and I came back with something that wasn't there before.",
]

# Additional evaluation stories - different from training prompts
EVAL_STORY_STARTERS = [
    "The elevator stopped between floors, and when the doors finally opened, I was standing in a hallway that didn't exist in the building plans.",
    "Every night at exactly 2:47 AM, my neighbor's dog starts barking at something in my backyard, but when I look, there's nothing there.",
    "The therapist told me to keep a dream journal, but I've never had the dream I keep writing about—the one where I'm watching myself sleep.",
    "The old woman at the bus stop smiled at me and said, 'You'll be joining us soon,' but when I turned around, she was gone and the bus stop was empty.",
    "My phone keeps receiving calls from my own number, and when I answer, I hear my own voice saying things I've never said.",
]

FULL_PROMPTS = [prepend_prompt + prompt for prompt in SCARY_STORY_STARTERS]
EVAL_PROMPTS = [prepend_prompt + prompt for prompt in EVAL_STORY_STARTERS]





def generate_completions(model, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.9, num_generations: int = 1) -> Dict[str, List[str]]:
    """
    Generate text completions for a single prompt using multinomial sampling.
    Can generate multiple completions in parallel for the same prompt.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding text
        prompt: Single prompt string to complete
        max_tokens: Maximum number of tokens to generate per completion
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
        num_generations: Number of parallel completions to generate (batch size)
    
    Returns:
        Dict with 'full_text' and 'completions' keys, each containing lists of strings
    """
    model.eval()
    
    with torch.no_grad():
        # Tokenize the prompt once
        input_ids = tokenizer._tokenizer.encode(prompt, allowed_special="all")
        prompt_length = len(input_ids)
        
        # Create batch by repeating the prompt
        batch_input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).repeat(num_generations, 1).cuda()
        current_ids = batch_input_ids.clone()
        
        # Generate tokens one by one with a progress bar
        for _ in trange(max_tokens, desc="Generating tokens", leave=False):
            # Get model predictions with autocast for bfloat16
            logits, _ = model(current_ids)
            
            logits = logits[:, -1, :]  # Get logits for last token
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Convert to probabilities and sample
            probs = F.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append the new tokens
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
            
            # Clean up memory
            del logits
        
        # Decode all generated sequences
        full_texts = []
        completions = []
        for i in range(num_generations):
            full_text = tokenizer.decode(current_ids[i].tolist())
            # Extract just the completion part (everything after the original prompt)
            completion_tokens = current_ids[i][prompt_length:].tolist()
            completion_text = tokenizer.decode(completion_tokens)
            
            full_texts.append(full_text)
            completions.append(completion_text)
    
    return {
        "full_text": full_texts,
        "completions": completions
    }


def generate_completions_nano(prompt: str, max_tokens: int = 100, temperature: float = 0.9, num_generations: int = 1) -> Dict[str, List[str]]:
    """
    Generate text completions for a single prompt using gpt-4.1-nano.
    Can generate multiple completions in parallel for the same prompt.
    
    Args:
        prompt: Single prompt string to complete
        max_tokens: Maximum number of tokens to generate per completion
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
        num_generations: Number of parallel completions to generate (batch size)
    
    Returns:
        Dict with 'full_text' and 'completions' keys, each containing lists of strings
    """
    client = OpenAI()
    
    full_texts = []
    completions = []
    
    for _ in range(num_generations):
        system_message = "I'm going to give you a scenario. Respond just as if you are continuing the text. Don't say anything else as if you are talking to a user - just continue the story naturally."
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        generated_text = response.choices[0].message.content
        full_text = prompt + generated_text
        
        full_texts.append(full_text)
        completions.append(generated_text)
    
    return {
        "full_text": full_texts,
        "completions": completions
    }



def story_v_story(story_1, story_2, max_retries=3): 
    """
    Takes two scary stories, prompts gpt to tell which is the best, weighting most of all coherence, but then scaryness. 

    Returns dict with story_1_better (bool) and reasoning (str)
    """
    class StoryComparison(BaseModel):
        story_1_better: bool
        reasoning: str

    client = OpenAI()
    
    # Randomly swap stories to avoid bias
    swapped = random.choice([True, False])
    if swapped:
        story_1, story_2 = story_2, story_1
    
    prompt = f"""You are an expert judge of scary stories. Compare these two stories and determine which one is better.

            Evaluation criteria (in order of importance):
            1. Coherence and narrative flow
            2. Scariness and horror elements  
            3. Writing quality and style

            Story 1:
            {story_1}

            Story 2:
            {story_2}

            Which story is better? You must pick the best option even if both stories are bad. Set story_1_better to True if Story 1 is better, or False if Story 2 is better. Provide your reasoning."""

    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": "You are an expert judge of scary stories. Compare stories based on coherence, scariness, and writing quality."},
                    {"role": "user", "content": prompt}
                ],
                text_format=StoryComparison,
            )
            
            result = response.output_parsed
            
            # If we swapped, flip the result back
            if swapped:
                result.story_1_better = not result.story_1_better
            
            return {
                "story_1_better": result.story_1_better,
                "reasoning": result.reasoning
            }
            
        except Exception as e:
            print(f"Judgment attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, return a random result
                print("All judgment attempts failed, using random result")
                return {
                    "story_1_better": random.choice([True, False]),
                    "reasoning": "Random fallback due to API error"
                }
            # Wait before retry
            import time
            time.sleep(1)


def evaluate_model(model, tokenizer, output_dir: str, iteration: int, max_tokens: int) -> float:
    """
    Evaluate the model against gpt-4.1-nano by comparing story completions.
    
    Args:
        model: The local model to evaluate
        tokenizer: Tokenizer for the local model
        output_dir: Directory to save evaluation results
        iteration: Iteration number for file naming
    
    Returns:
        Win rate of the local model (0.0 to 1.0)
    """
    print(f"Starting evaluation iteration {iteration}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters
    total_comparisons = 0
    local_model_wins = 0
    
    # Store detailed results
    evaluation_results = {
        "iteration": iteration,
        "total_prompts": len(EVAL_PROMPTS),
        "comparisons_per_prompt": 3,
        "total_comparisons": len(EVAL_PROMPTS) * 3,
        "local_model_wins": 0,
        "win_rate": 0.0,
        "detailed_results": []
    }
    
    # Iterate through each prompt
    for prompt_idx, prompt in enumerate(tqdm(EVAL_PROMPTS, desc="Evaluating prompts")):
        print(f"\nEvaluating prompt {prompt_idx + 1}/{len(EVAL_PROMPTS)}")
        
        # Generate 3 completions from both models
        print("Generating local model completions...")
        local_results = generate_completions(model, tokenizer, prompt, max_tokens=max_tokens, num_generations=3)
        
        print("Generating gpt-4.1-nano completions...")
        nano_results = generate_completions_nano(prompt, max_tokens=max_tokens, num_generations=3)
        
        # Compare each pair
        for comp_idx in range(3):
            local_completion = local_results["completions"][comp_idx]
            nano_completion = nano_results["completions"][comp_idx]
            
            print(f"Comparing completion {comp_idx + 1}/3...")
            
            # Get judgment
            judgment = story_v_story(local_completion, nano_completion)
            
            # Determine winner
            local_wins = judgment["story_1_better"]  # True if local model wins
            
            if local_wins:
                local_model_wins += 1
            
            total_comparisons += 1
            
            # Store detailed result
            comparison_result = {
                "prompt_index": prompt_idx,
                "prompt": prompt,
                "comparison_index": comp_idx,
                "local_model_completion": local_completion,
                "nano_completion": nano_completion,
                "local_model_wins": local_wins,
                "judge_reasoning": judgment["reasoning"]
            }
            
            evaluation_results["detailed_results"].append(comparison_result)
            
            print(f"Local model wins: {local_wins}")
            print(f"Current win rate: {local_model_wins}/{total_comparisons} = {local_model_wins/total_comparisons:.3f}")

    # Calculate final win rate
    win_rate = local_model_wins / total_comparisons if total_comparisons > 0 else 0.0
    
    # Update results
    evaluation_results["local_model_wins"] = local_model_wins
    evaluation_results["win_rate"] = win_rate
    
    # Save results to JSON file
    output_file = os.path.join(output_dir, f"eval_metrics_{iteration}.json")
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Local model win rate: {win_rate:.3f} ({local_model_wins}/{total_comparisons})")
    print(f"Results saved to: {output_file}")
    
    return win_rate


def generate_completions_and_scores(model, tokenizer, prompt: str, output_dir: str, iteration: int, max_tokens: int, num_generations: int = 8) -> tuple:
    """
    Generate completions and score them through round-robin competition.
    
    Args:
        model: The local model to use for generation
        tokenizer: Tokenizer for the local model
        prompt: Single prompt to complete
        output_dir: Directory to save results
        iteration: Iteration number for file naming
        num_generations: Number of completions to generate (default 8)
    
    Returns:
        Tuple of (full_texts, completions, win_rates) - all lists in order
    """
    print(f"Generating {num_generations} completions and scoring...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate completions
    print("Generating completions...")
    results = generate_completions(model, tokenizer, prompt, max_tokens=max_tokens, num_generations=num_generations)
    full_texts = results["full_text"]
    completions = results["completions"]
    
    # Initialize scoring
    story_scores = {i: {"wins": 0, "competitions": 0, "win_rate": 0.0} for i in range(num_generations)}
    all_competitions = []
    
    # Round-robin competition
    print("Running round-robin competitions...")
    total_comparisons = 0
    
    for i in range(num_generations):
        for j in range(i + 1, num_generations):
            story_1 = completions[i]
            story_2 = completions[j]
            
            print(f"Comparing story {i+1} vs story {j+1}...")
            
            # Get judgment
            judgment = story_v_story(story_1, story_2)
            
            # Update scores
            story_scores[i]["competitions"] += 1
            story_scores[j]["competitions"] += 1
            
            if judgment["story_1_better"]:
                story_scores[i]["wins"] += 1
                winner = i
                loser = j
            else:
                story_scores[j]["wins"] += 1
                winner = j
                loser = i
            
            total_comparisons += 1
            
            # Record competition
            competition = {
                "story_1_index": i,
                "story_2_index": j,
                "story_1_text": story_1,
                "story_2_text": story_2,
                "winner": winner,
                "judge_result": judgment["story_1_better"],
                "judge_reasoning": judgment["reasoning"]
            }
            all_competitions.append(competition)
    
    # Calculate win rates
    win_rates = []
    for i in range(num_generations):
        if story_scores[i]["competitions"] > 0:
            win_rate = story_scores[i]["wins"] / story_scores[i]["competitions"]
        else:
            win_rate = 0.0
        story_scores[i]["win_rate"] = win_rate
        win_rates.append(win_rate)
    
    # Prepare results for JSON
    results_data = {
        "iteration": iteration,
        "prompt": prompt,
        "num_generations": num_generations,
        "total_comparisons": total_comparisons,
        "story_scores": story_scores,
        "competitions": all_competitions
    }
    
    # Save to JSON
    output_file = os.path.join(output_dir, f"completions_scores_{iteration}.json")
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"Win rates: {[f'{wr:.3f}' for wr in win_rates]}")
    
    return full_texts, completions, win_rates


def selective_log_softmax(logits, index):
    """
    Memory-efficient log_softmax -> gather operation.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    """
    Get per-token log probabilities for the completion tokens.
    """
    # Get logits from model (just pass input_ids as positional argument)
    logits, _ = model(input_ids)
    logits = logits[:, :-1, :]  # Remove last logit (next token prediction)
    
    # Take only the completion tokens
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    
    return selective_log_softmax(logits, input_ids)


def compute_grpo_loss(model, prompt_ids, completion_ids, attention_mask, advantages):
    """
    Compute GRPO loss for the generated completions.
    """
    # Reconstruct full input sequence
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    # Get per-token log probabilities for completion tokens only
    logps = get_per_token_logps(model, input_ids, attention_mask, completion_ids.size(1))
    
    # Compute GRPO loss: -exp(logp - logp.detach()) * advantages
    per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    
    # Average loss over batch and sequence length
    loss = per_token_loss.mean()
    
    return loss








def main() -> None:
    """
    Main function orchestrating story generation and evaluation pipeline.
    
    Handles command line arguments and executes the requested operations:
    1. Story generation using trained model (if --generate_stories)
    2. Story evaluation using GPT-4 (if --get_gpt_ranking)
    """
    parser = argparse.ArgumentParser(description='Generate and evaluate stories using trained GPT model')

    parser.add_argument('--output_folder', type=str, default='grpo_out', help='Output folder to save model and generated stories')
    parser.add_argument('--checkpoint_path', type=str, default='ghoulish_pretrained_terrifier', help='Path to model checkpoint')
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum tokens to generate per completion')
    
    # Training arguments
    parser.add_argument('--num_train_iters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate every N steps')
    parser.add_argument('--save_every', type=int, default=100, help='Save model every N steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--warmup_percent', type=float, default=0.1, help='Percentage of training steps for warmup')
    parser.add_argument('--num_generations', type=int, default=4, help='Number of completions to generate per training step')
    
    args = parser.parse_args()


    # Load the model 
    net = models.get_model("gpt2full")
    state_dict = torch.load(os.path.join(args.checkpoint_path,"checkpoint.pt"))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict['model_state_dict'].items()}
    net.load_state_dict(state_dict)
    net = net.cuda()
    net.eval()

    # Get tokenizer
    tokenizer = tokenizers.get_tokenizer("gpt2")
    
    # Run first evaluation at step 0 (baseline)
    print("Running initial evaluation...")
    # win_rate = evaluate_model(net, tokenizer, args.output_folder, iteration=0, max_tokens=args.max_tokens)
    # print(f"Initial model win rate: {win_rate:.3f}")
    
    # Setup training
    net.train()  # Set to training mode
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler with warmup
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / max(warmup_steps, 1))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    
    # Training log
    training_log = {
        "args": vars(args),
        "steps": {}
    }
    
    accumulated_loss = 0.0
    optimizer.zero_grad()
    
    print(f"Starting GRPO training for {args.num_train_iters} iterations...")
    
    # Training loop
    for step in tqdm(range(args.num_train_iters), desc="Training"):
        # Initialize step entry in training log
        if step not in training_log["steps"]:
            training_log["steps"][step] = {}
        
        # Periodic evaluation
        if step % args.eval_every == 0:
            print(f"\nEvaluating at step {step + 1}...")
            net.eval()
            eval_win_rate = evaluate_model(net, tokenizer, args.output_folder, iteration=step+1, max_tokens=args.max_tokens)
            net.train()
            
            training_log["steps"][step]["eval_win_rate"] = eval_win_rate
            print(f"Evaluation win rate: {eval_win_rate:.3f}")
        

        # Randomly select a prompt
        prompt = random.choice(FULL_PROMPTS)
        
        # Generate completions and get scores
        full_texts, completions, win_rates = generate_completions_and_scores(
            net, tokenizer, prompt, args.output_folder, iteration=step, 
            max_tokens=args.max_tokens, num_generations=args.num_generations
        )

        
        # Convert win rates to advantages (normalize around mean)
        device = next(net.parameters()).device
        win_rates_tensor = torch.tensor(win_rates, device=device, dtype=torch.float32)
        mean_win_rate = win_rates_tensor.mean()
        std_win_rate = win_rates_tensor.std()
        advantages = (win_rates_tensor - mean_win_rate) / (std_win_rate + 1e-4)

        # Tokenize all completions
        prompt_ids = tokenizer._tokenizer.encode(prompt, allowed_special="all")
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        # Repeat prompt to match batch size of completions
        prompt_tensor = prompt_tensor.repeat(len(completions), 1)
        
        # Tokenize completions
        completion_ids_list = []
        for completion in completions:
            completion_ids = tokenizer._tokenizer.encode(completion, allowed_special="all")
            completion_tensor = torch.tensor(completion_ids, dtype=torch.long, device=device)
            completion_ids_list.append(completion_tensor)
        
        # Pad completions to same length (since we know they're all the same length)
        max_completion_len = max(len(ids) for ids in completion_ids_list)
        completion_ids_batch = torch.zeros(len(completion_ids_list), max_completion_len, dtype=torch.long, device=device)
        
        for i, completion_ids in enumerate(completion_ids_list):
            completion_ids_batch[i, :len(completion_ids)] = completion_ids
        
        # Create attention mask
        attention_mask = torch.ones(prompt_tensor.size(0), prompt_tensor.size(1) + completion_ids_batch.size(1), device=device)
        
        # Compute GRPO loss
        loss = compute_grpo_loss(net, prompt_tensor, completion_ids_batch, attention_mask, advantages)

        # Backward pass
        (loss / args.gradient_accumulation_steps).backward()
        accumulated_loss += loss.item()

        # Optimizer step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        
        # Log training step (update existing entry to preserve eval data)
        training_log["steps"][step].update({
            "prompt": prompt,
            "completions": completions,
            "win_rates": win_rates,
            "advantages": advantages.tolist(),
            "loss": loss.item(),
            "accumulated_loss": accumulated_loss,
            "lr": scheduler.get_last_lr()[0]
        })
        

        # Periodic model saving
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_folder, f"checkpoint_step_{step+1}.pt")
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step + 1,
                'loss': accumulated_loss
            }, checkpoint_path)
            print(f"Saved checkpoint at step {step+1}")
        
        # Save training log
        with open(os.path.join(args.output_folder, "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2)
    
    print("Training complete!")
    



if __name__ == '__main__':
    main()
