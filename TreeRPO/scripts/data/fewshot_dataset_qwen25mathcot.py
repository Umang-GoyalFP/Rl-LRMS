"""Script to prepare MATH training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing MATH models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
import json
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str, data_source: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, instruction: str = None) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        answer = example.pop('answer')

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{{}}."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for MATH training')
    parser.add_argument('--local_dir', default=os.path.expanduser('/hpc2hdd/home/zyang398/yangzhch6/projs/TreeRPO/TreeRPO-v4/data_qwen25_math_cot'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    # Initialize datasets
    # print([item.value for item in TrainDataset.Math])
    # train_datasets = [TrainDataset.Math.MATH]
    # train_dataset = load_dataset(train_datasets[0])
    # print(type(train_dataset[0]))
    # exit()

    path = "/hpc2hdd/home/zyang398/yangzhch6/projs/TreeRPO/TreeRPO-v4/rllm/data/train/math/math_cot.json"

    # load json
    with open(path, 'r') as f:
        train_dataset = json.load(f)[:16]

    print(len(train_dataset))
    # exit()

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn(split='train', data_source="FewShot")
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # random shuffle train_data
    import random
    seed = 42
    random.seed(seed)
    random.shuffle(train_data)

    # Save dataset parquet
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train/fewshot.parquet'))
    
    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)