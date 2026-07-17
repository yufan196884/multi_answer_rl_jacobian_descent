import json
from itertools import islice

path = "/home/yufan/data_101/multi_answer_rl_jacobian_descent/data/maze_smoke/train.jsonl"

with open(path, "r", encoding="utf-8") as file:
    for index, line in enumerate(islice(file, 5)):
        example = json.loads(line)

        print(f"\nMaze {index} | seed={example['seed']}")
        print("\n".join(example["grid"]))