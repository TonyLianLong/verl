import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(example):
    example['answer'] = remove_boxed(example['answer'])
    return example


def get_solution(example):
    example['answer'] = example['answer'][0]
    return example


# add a row to each data item that represents a unique id
def make_map_fn(split, data_source):
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example.pop("problem")
        question = question + " " + instruction_following

        solution = example.pop("answer")
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx},
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.expanduser("~/verl/data/mathmix"))
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # load training dataset
    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # process training dataset
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)

    # load test dataset
    data_sources = ['math-ai/math500', 'math-ai/minervamath', 'math-ai/olympiadbench', 'math-ai/amc23', 'math-ai/aime24', 'math-ai/aime25']
    test_dataset = []
    test_dataset.append(datasets.load_dataset('math-ai/math500', split='test').select_columns(['problem', 'answer']))
    test_dataset.append(datasets.load_dataset('math-ai/minervamath', split='test').rename_column("question", "problem").select_columns(['problem', 'answer']))
    test_dataset.append(datasets.load_dataset('math-ai/olympiadbench', split='test').rename_column("question", "problem").rename_column("final_answer", "answer").select_columns(['problem', 'answer']).map(get_solution))
    test_dataset.append(datasets.load_dataset('math-ai/amc23', split='test').rename_column("question", "problem").select_columns(['problem', 'answer']))
    test_dataset.append(datasets.load_dataset('math-ai/aime24', split='test').rename_column("solution", "answer").select_columns(['problem', 'answer']).map(extract_solution))
    test_dataset.append(datasets.load_dataset('math-ai/aime25', split='test').select_columns(['problem', 'answer']))

    # process test dataset
    for i in range(len(test_dataset)):
        test_dataset[i] = test_dataset[i].map(function=make_map_fn("test", data_sources[i]), with_indices=True)
    test_dataset = datasets.concatenate_datasets(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
