import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.utils.torch_functional import pad_sequence_to_length

class TreeNode:
    def __init__(self, batch_dict={}, depth=0, prompt_length=512, max_response_length=1280, step_length=256, reward=0, eos_token_id=None):
        self.depth = depth
        self.prompt_length = prompt_length
        self.max_response_length = max_response_length
        self.step_length = step_length

        self.E_reward = reward
        self.children = []
        self.child_rewards = []

        self.batch_dict = batch_dict
        self.eos_token_id = eos_token_id
        self.is_end = False

        assert self.batch_dict["input_ids"].shape[1] == self.prompt_length + self.depth * self.step_length
        assert self.batch_dict["input_ids"].shape == self.batch_dict["attention_mask"].shape
        assert self.batch_dict["input_ids"].shape == self.batch_dict["position_ids"].shape
        assert self.batch_dict["input_ids"].shape[0] == 1

        if depth != 0:
            valid_response_length = self.batch_dict["attention_mask"][0, -self.step_length:].sum().item()
            self.is_end = (self.batch_dict["responses"][0, valid_response_length - 1] == self.eos_token_id).item()
    
    def add_child(self, child_node):
        self.children.append(child_node)

    def format_node_input_batch(self):
        node_input_batch: DataProto = DataProto.from_single_dict({
            'input_ids': self.batch_dict['input_ids'],
            'attention_mask': self.batch_dict['attention_mask'],
            'position_ids': self.batch_dict['position_ids'],
            'data_source': self.batch_dict['data_source'],
            'ability': self.batch_dict['ability'],
            'reward_model': self.batch_dict['reward_model'],
            'extra_info': self.batch_dict['extra_info'],
            'index': self.batch_dict['index']
        })
        return node_input_batch
    
    # need padding
    def format_node_train_batch(self, pad_token_id):
        max_depth = self.max_response_length // self.step_length
        assert self.max_response_length % self.step_length == 0
        left_pad_max_length = self.prompt_length + max_depth * self.step_length

        node_input_batch: DataProto = DataProto.from_single_dict({
            'input_ids': pad_sequence_to_length(self.batch_dict['input_ids'][0], left_pad_max_length, pad_token_id, left_pad=True).unsqueeze(0),
            'attention_mask': pad_sequence_to_length(self.batch_dict['attention_mask'][0], left_pad_max_length, 0, left_pad=True).unsqueeze(0),
            'position_ids': pad_sequence_to_length(self.batch_dict['position_ids'][0], left_pad_max_length, 0, left_pad=True).unsqueeze(0),
            'prompts': pad_sequence_to_length(self.batch_dict['prompts'][0], left_pad_max_length-self.step_length, pad_token_id, left_pad=True).unsqueeze(0),
            'responses': self.batch_dict['responses'],
            'token_level_rewards': self.batch_dict['token_level_rewards'],
            'token_level_scores': self.batch_dict['token_level_scores'],
            'advantages': self.batch_dict['advantages'],
            'returns': self.batch_dict['returns'],
            'data_source': self.batch_dict['data_source'],
            'ability': self.batch_dict['ability'],
            'reward_model': self.batch_dict['reward_model'],
            'extra_info': self.batch_dict['extra_info'],
            'index': self.batch_dict['index']
        })
        return node_input_batch

    def print_node(self):
        print(f"Node depth: {self.depth}")
        print(f"Node is_end: {self.is_end}")
        print(f"Node children: {len(self.children)}")
        print(f"Node E_reward: {self.E_reward}")
        print(f"Node batch_dict: {self.batch_dict}")

