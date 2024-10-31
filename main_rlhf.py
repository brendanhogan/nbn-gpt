"""
RLHF Story Generation and Evaluation Script

This script implements a pipeline for generating and evaluating stories using a combination of
GPT-like language models and GPT-4 based ranking. It supports:

- Story generation using a trained GPT model checkpoint
- Automated story evaluation and ranking using GPT-4
- Distributed story generation in batches
- Persistent storage of generated stories and rankings
- Robust error handling and retry logic for API calls

The evaluation process includes:
- Grouping stories into sets of 4 for comparative ranking
- Getting GPT-4 judgments on story quality and scariness
- Saving best stories and rankings for further analysis

Example:
    $ python main_rlhf.py --checkpoint_path model.pt --output_folder output/ --generate_stories --get_gpt_ranking

"""
import json
import os
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Optional

from generate_stories import generate_stories


def get_gpt_ranking(stories: Dict[str, str], output_folder: str) -> Dict[str, Dict]:
    """
    Get GPT-4 rankings for groups of stories with exponential backoff retry.
    
    Args:
        stories: Dictionary mapping story IDs to story text
        output_folder: Directory to save ranking results
        
    Returns:
        Dictionary containing rankings and metadata for each story group
        
    The function:
    1. Groups stories into sets of 4
    2. Gets GPT-4 judgment on the scariest story in each group
    3. Saves rankings incrementally with error handling
    4. Returns complete rankings dictionary
    """
    rankings_file = os.path.join(output_folder, 'gpt_rankings.json')
    
    # Load existing rankings if they exist
    if os.path.exists(rankings_file):
        with open(rankings_file, 'r') as f:
            rankings = json.load(f)
    else:
        rankings = {}

    # Initialize OpenAI client
    client = OpenAI()

    # Process stories in groups of 4
    story_ids = list(stories.keys())
    for i in tqdm(range(0, len(story_ids), 4), desc="Getting GPT-4 rankings"):
        group = story_ids[i:i+4]
        group_key = f"group_{i//4}"
        
        # Skip if we already have ranking for this group
        if group_key in rankings:
            continue
            
        # Map stories A-D for GPT-4 prompt
        story_dict = {chr(65 + j): stories[story_id] for j, story_id in enumerate(group)}
        
        # Construct prompt for GPT-4
        prompt = (
            "Please read these 4 stories and tell me which one is the scariest. "
            "First respond with BEST_ANSWER:[X] where X is A, B, C, or D. "
            "Then on a new line write REASONING:[] followed by a clear explanation "
            "of why you chose that story as the scariest. It is extremely important "
            "you answer both in brackets so that we can parse the answer in python.\n\n"
        )
        for label, story in story_dict.items():
            prompt += f"Story {label}:\n{story}\n\n"

        # Call GPT-4 with exponential backoff retry
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Make API call to GPT-4
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert story reader, and an expert at knowing what makes for the scariest stories"
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                )
                
                # Parse and validate response
                response_text = response.choices[0].message.content
                if "BEST_ANSWER:[" in response_text and "REASONING:" in response_text:
                    best = response_text.split("BEST_ANSWER:[")[1].split("]")[0]
                    reasoning = response_text.split("REASONING:")[1].strip()
                    
                    # Store ranking results
                    rankings[group_key] = {
                        "stories": {chr(65+j): story_id for j, story_id in enumerate(group)},
                        "best": best,
                        "reasoning": reasoning,
                        "best_story_text": stories[group[ord(best) - 65]]
                    }
                    
                    # Save incrementally
                    with open(rankings_file, 'w') as f:
                        json.dump(rankings, f, indent=2)
                        
                    print(f"Completed {group_key}")
                    break  # Success - exit retry loop
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    raise  # Re-raise the last exception
                    
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
    return rankings

def main() -> None:
    """
    Main function orchestrating story generation and evaluation pipeline.
    
    Handles command line arguments and executes the requested operations:
    1. Story generation using trained model (if --generate_stories)
    2. Story evaluation using GPT-4 (if --get_gpt_ranking)
    """
    parser = argparse.ArgumentParser(description='Generate and evaluate stories using trained GPT model')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder to save generated stories and json')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--generate_stories', action='store_true',
                       help='Generate stories using the trained model')
    parser.add_argument('--get_gpt_ranking', action='store_true',
                       help='Get GPT-4 rankings for generated stories')
    args = parser.parse_args()

    if args.generate_stories:
        # Generate stories using model
        stories = generate_stories(
            model_type='gpt2full',
            output_dir=os.path.join(args.output_folder, 'stories'),
            checkpoint_path=args.checkpoint_path,
            num_stories=400,
            max_tokens=300,
            batch_size=100
        )

        # Save stories to JSON
        stories_dict = {f"story_{i}": story for i, story in enumerate(stories)}
        os.makedirs(args.output_folder, exist_ok=True)
        json_path = os.path.join(args.output_folder, 'generated_stories.json')
        
        with open(json_path, 'w') as f:
            json.dump(stories_dict, f, indent=2)

        print(f"\nGenerated 400 stories and saved to:")
        print(f"Individual files: {os.path.join(args.output_folder, 'stories')}")
        print(f"Combined JSON: {json_path}")

    if args.get_gpt_ranking:
        # Load generated stories
        json_path = os.path.join(args.output_folder, 'generated_stories.json')
        with open(json_path, 'r') as f:
            stories = json.load(f)
            
        # Get and save rankings
        rankings = get_gpt_ranking(stories, args.output_folder)
        print(f"\nRankings saved to: {os.path.join(args.output_folder, 'gpt_rankings.json')}")
        
        # Save best stories to text file
        best_stories_path = os.path.join(args.output_folder, 'best_stories.txt')
        with open(best_stories_path, 'w') as f:
            for group, data in rankings.items():
                f.write(data['best_story_text'])
                f.write("\n\n")
        
        print(f"Best stories saved to: {best_stories_path}")

if __name__ == '__main__':
    main()
