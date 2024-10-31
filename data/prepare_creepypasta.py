
import numpy as np 
import pandas as pd 
import os
import re
from tqdm import tqdm

# Load file 
creepypastas = pd.read_excel("creepypastas.xlsx")

# Create a subdirectory called 'creepypasta_stories'
os.makedirs('creepypasta_stories', exist_ok=True)

# Initialize an empty string to store all stories
all_stories = ""

# Iterate through the DataFrame and write each story to a text file in the subdirectory
for idx, row in tqdm(creepypastas.iterrows(), total=len(creepypastas), desc="Processing stories"):
    story_name = row['story_name']
    body = row['body']
    # Create a valid filename
    tmp = re.sub(r'[^\w\-_. ]', '', story_name.replace(' ', '_'))
    filename = f"{idx}_{tmp}.txt"
    # Write the body to a text file in the subdirectory
    with open(os.path.join('creepypasta_stories', filename), 'w', encoding='utf-8') as file:
        file.write(body)
    
    # Append the story to all_stories
    all_stories += body + "\n\n"

# Trim down characters for main dataset 
chars = sorted(list(set(all_stories)))
# Only allows alpha numerics, and some punctuation 
allowed_chars = chars[:96] 
# Replace characters not in allowed_chars with a space
all_stories = ''.join(char if char in allowed_chars else ' ' for char in all_stories)
# Remove consecutive spaces
all_stories = ' '.join(all_stories.split())

print(f"Created {len(creepypastas)} text files in the 'creepypasta_stories' directory.")

# Write all stories to a single file
with open('all_stories.txt', 'w', encoding='utf-8') as file:
    file.write(all_stories)

print("Created 'all_stories.txt' containing all stories.")
















#