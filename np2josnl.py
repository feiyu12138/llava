import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--score-file', type=str)
args = parser.parse_args()
score = np.load(args.score_file, allow_pickle=True).tolist()

# output to jsonl file
with open(args.score_file.replace('.npy', '.jsonl'), 'w') as f:
    for s in score:
        f.write(json.dumps(s) + '\n')    
    