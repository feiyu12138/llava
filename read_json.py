import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage
file_path = '/home/lye21/LLaVA/playground/data/LLaVA-Tuning/llava_v1_5_mix665k.json'
json_data = read_json_file(file_path)
print(len(json_data))