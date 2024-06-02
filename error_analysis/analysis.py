import json
import os
file1='stride-16-layer-1-grouping-avgpool1d-retrain.jsonl'
file2='1dpool16layer16_wortrain.jsonl'
file3='1dpool4layer16-7b.jsonl'
file4='llava-v1.5-7b-eval2.jsonl'
file5='1dpool4layer16_v3.jsonl'
samples1 = []
samples2 = []
samples3 = []
samples4 = []
samples5 = []
ocr_count = 0
know_count = 0
know_num = 0
ocr = [15, 16, 36, 37, 38, 39, 46, 47, 48, 49, 55, 56, 57]
#read jsonl file line by line
with open(file1, 'r') as f:
    for line in f:
        samples1.append(json.loads(line))
with open(file2, 'r') as f:
    for line in f:
        samples2.append(json.loads(line))
with open(file3, 'r') as f:
    for line in f:
        samples3.append(json.loads(line))
with open(file4, 'r') as f:
    for line in f:
        samples4.append(json.loads(line))
with open(file5, 'r') as f:
    for line in f:
        samples5.append(json.loads(line))
#compare the samples
samples = samples4
print("ocr ratio:", len(ocr)/len(samples1))
for i in range(len(samples1)):
    if samples[i]['category'] == 'llava_bench_conv':
        know_num += 1 
    score1 = samples[i]['tuple'][1]
    score2 = samples[i]['tuple'][0]
    if score1/score2 >= 0.8:
        print(f'{i} {score1} < {score2}: category: {samples1[i]["category"]}, ocr: {samples1[i]["id"] in ocr}')
        if samples1[i]["id"] in ocr:
            ocr_count += 1
        if samples4[i]['category'] == 'llava_bench_conv':
            know_count += 1
print("ocr count:", ocr_count / len(ocr))
print("know count:", know_count / know_num)