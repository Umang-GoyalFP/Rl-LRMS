# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import json
import wandb
import uuid
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from tqdm import tqdm
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from rllm.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from verl.trainer.ppo import tree
from evals.fitness_utils import (
    compute_tree_fitness,
    classify_tree,
    collect_leaf_rewards,
    collect_node_logprobs_rewards_from_batch,
)

WorkerType = Type[Worker]


def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, mask_truncated_samples=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        mask_truncated_samples=mask_truncated_samples)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        train_batch_size = self.config.data.train_batch_size
        if self.config.trainer.rejection_sample:
            train_batch_size *= self.config.trainer.rejection_sample_multiplier
            train_batch_size = int(train_batch_size)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores, response_lengths, data_sources, experiment_name):
        """save validation generations to json file"""
        # save as json file
        json_folder = os.path.join(self.config.trainer.default_local_dir, "val_generations")
        save_data = []
        for line_input, line_output, line_score, line_response_length, line_data_source in zip(inputs, outputs, scores, response_lengths, data_sources):
            save_data.append({
                'input': line_input,
                'output': line_output,
                'score': line_score,
                'response_length': line_response_length.item(),
                'data_source': line_data_source,
            })
        json_file = os.path.join(json_folder, f"{self.global_steps}.json")
        os.makedirs(json_folder, exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(save_data, f, indent=4)


        """Log a table of validation samples to wandb"""

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first 1 samples after shuffling
        samples = samples[:1]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val_generations/{}".format(experiment_name): new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        response_length_lst = []
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # Store original inputs
            input_ids = test_gen_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('Validation: Generation end.')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            response_info = _compute_response_info(test_batch)
            response_length_lst.append(response_info['response_length'])

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        response_lengths = torch.cat(response_length_lst, dim=0).cpu() 
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, response_lengths=response_lengths, data_sources=data_sources, experiment_name=self.config.trainer.experiment_name)

        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_length = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_length[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_length[data_source].append(response_lengths[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            correct_count = np.sum(np.array(rewards) == RewardConfig.correct_reward)

            # compute acc score
            metric_dict[f'val/acc_score/{data_source}'] = float(correct_count) / len(rewards)

            # compute response avg length
            metric_dict[f'val_global_response_length_mean/{data_source}'] = np.mean(data_source_length[data_source])

            # compute correct response avg length
            is_correct = np.array(rewards) == RewardConfig.correct_reward
            correct_response_length = np.array(data_source_length[data_source])[is_correct]
            metric_dict[f'val_correct_response_length_mean/{data_source}'] = np.mean(correct_response_length)
            
            # compute incorrect response avg length
            incorrect_response_length = np.array(data_source_length[data_source])[~is_correct]
            metric_dict[f'val_incorrect_response_length_mean/{data_source}'] = np.mean(incorrect_response_length)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout',
                                                     reward_config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref',
                                                  reward_config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint_huggingface(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor_huggingface')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor_huggingface')
        self.actor_rollout_wg.save_checkpoint_higgingface(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic_huggingface')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic_huggingface')
            self.critic_wg.save_checkpoint_higgingface(critic_local_path, critic_remote_path)

    def _del_last_checkpoint(self):
        last_save_step = self.global_steps - self.config.trainer.save_freq

        # skip the test checkpoint
        if last_save_step % self.config.trainer.test_freq == 0:
            return

        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{last_save_step}')
        # del the folder
        if os.path.exists(local_global_step_folder):
            import shutil
            shutil.rmtree(local_global_step_folder)
            print(f'Deleted checkpoint folder: {local_global_step_folder}')

    def _save_checkpoint(self):
        self._save_checkpoint_huggingface()

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=False)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=False)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
        
        # del the last checkpoint, if 
        self._del_last_checkpoint()

    def _load_checkpoint(self):
        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if global_step_folder is None:
            print('Training from scratch')
            return 0

        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=False)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=False)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        from verl.utils.dataset.rl_dataset import RLHFDataset
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)



    def _format_train_batch(self, level_node_list):
        train_batch_list = []

        # format the input batch
        for target_depth in range(1, self.max_tree_depth + 1):
            if len(level_node_list[target_depth]) == 0:
                break

            for node in level_node_list[target_depth]:
                # pruning: if the reward change is too small, we shall skip this node
                if max(node.bro_rewards) - min(node.bro_rewards) < 0.1:
                    # print('pruning node...')
                    continue
                
                train_batch_list.append(node.format_node_train_batch(pad_token_id=self.tokenizer.pad_token_id))
        
        # assert len(train_batch_list) > 0
        if len(train_batch_list) == 0:
            return None
        train_batch = DataProto.concat(train_batch_list)
        return train_batch
    


    def _traversal_adv_(self, node, bro_rewards=[]):
        node.bro_rewards = bro_rewards

        if node.depth != 0:
            # give token level reward & scores
            node.batch_dict['token_level_scores'] = torch.zeros_like(node.batch_dict['responses'], dtype=torch.float32)
            response_length = node.batch_dict['responses'].shape[-1]
            final_index = node.batch_dict['attention_mask'][0, -response_length:].sum(-1).item() - 1
            node.batch_dict['token_level_scores'][0, final_index] = node.E_reward
            node.batch_dict['token_level_rewards'] = node.batch_dict['token_level_scores']

            # compute adv
            mean_reward = np.mean(node.bro_rewards)
            std_reward = mean_reward * (1 - mean_reward)

            node_adv = (node.E_reward - mean_reward) / (std_reward + 1e-6)
            node_adv_tensor = torch.full((1, response_length), node_adv, dtype=torch.float32)
            node_adv_tensor *= node.batch_dict['attention_mask'][:, -response_length:]
            node.batch_dict['advantages'] = node_adv_tensor
            node.batch_dict['returns'] = node_adv_tensor

        if node.is_end or node.depth == self.max_tree_depth:
            return 
        for child in node.children:
            self._traversal_adv_(child, bro_rewards=node.child_rewards)


    def _traversal_adv(self, root_list):
        for root in root_list:
            assert not root.is_end
            self._traversal_adv_(root, [])


    def _traversal_reward_(self, node):
        if node.is_end or node.depth == self.max_tree_depth:
            return node.E_reward

        assert len(node.children) == self.config.actor_rollout_ref.rollout.n
        reward = 0
        for child in node.children:
            current_child_reward = self._traversal_reward_(child)
            reward += current_child_reward
            node.child_rewards.append(current_child_reward)

        assert len(node.children) > 0
        node.E_reward = reward / len(node.children)
        return node.E_reward

 
    def _traversal_reward(self, root_list):
        for root in root_list:
            assert not root.is_end
            self._traversal_reward_(root)


    def _format_child_node_(self, node, father_depth, gen_batch, idx, reward):
        if node.is_end or father_depth == self.max_tree_depth:
            return
        
        start_idx = self.config.data.max_prompt_length + self.config.actor_rollout_ref.rollout.step_length * father_depth
        end_idx = start_idx + self.config.actor_rollout_ref.rollout.step_length

        assert torch.equal(gen_batch.batch['input_ids'][idx, :gen_batch.batch['prompts'].shape[1]], gen_batch.batch['prompts'][idx])
        assert torch.equal(gen_batch.batch['input_ids'][idx, -gen_batch.batch['responses'].shape[1]:], gen_batch.batch['responses'][idx])

        node.add_child(
            tree.TreeNode(
                batch_dict={
                    'input_ids': gen_batch.batch['input_ids'][idx:idx+1, :end_idx],
                    'attention_mask': gen_batch.batch['attention_mask'][idx:idx+1, :end_idx],
                    'position_ids': gen_batch.batch['position_ids'][idx:idx+1, :end_idx],
                    'prompts': gen_batch.batch['input_ids'][idx:idx+1, :start_idx],
                    'responses': gen_batch.batch['input_ids'][idx:idx+1, start_idx:end_idx],
                    'data_source': gen_batch.non_tensor_batch['data_source'][idx:idx+1],
                    'ability': gen_batch.non_tensor_batch['ability'][idx:idx+1],
                    'reward_model': gen_batch.non_tensor_batch['reward_model'][idx:idx+1],
                    'extra_info': gen_batch.non_tensor_batch['extra_info'][idx:idx+1],
                    'index': gen_batch.non_tensor_batch['index'][idx:idx+1],
                },
                depth=father_depth+1,
                prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                step_length=self.config.actor_rollout_ref.rollout.step_length,
                reward=reward[idx].item(),
                eos_token_id=self.tokenizer.eos_token_id
            )
        )
        self._format_child_node_(node.children[-1], father_depth+1, gen_batch, idx, reward)


    def _format_child_node(self, level_node_list, current_depth, gen_batch, reward, pad_size):
        idx = 0
        for node in level_node_list[current_depth]:
            if node.is_end:
                continue
            for k in range(gen_batch.meta_info['sample_n'] * idx, gen_batch.meta_info['sample_n'] * (idx + 1)):
                self._format_child_node_(node, current_depth, gen_batch, k, reward)
            
            for child in node.children:
                node.E_reward += child.E_reward
            node.E_reward /= len(node.children)

            idx += 1

        assert idx * gen_batch.meta_info['sample_n'] == gen_batch.batch['input_ids'].shape[0] - pad_size * gen_batch.meta_info['sample_n']
        assert reward.shape[0] == gen_batch.batch['input_ids'].shape[0]

    def _format_depth_input_batch(self, target_depth_node_list):
        depth_batch_list = []
        for node in target_depth_node_list:
            if node.is_end:
                continue
            depth_batch_list.append(node.format_node_input_batch())

        pad_size = 0
        if len(depth_batch_list) == 0:
            return None, pad_size
        
        depth_batch = DataProto.concat(depth_batch_list)
        print("** depth batch size:", depth_batch.batch['input_ids'].shape[0])

        depth_batch, pad_size = pad_dataproto_to_divisor(depth_batch, self.actor_rollout_wg.world_size)

        return depth_batch, pad_size

    # ===================================================================
    # CATPO: Fitness computation and critique-guided healing
    # ===================================================================

    def _compute_tree_fitness_all(self, level_node_list, train_batch):
        """Compute fitness F(T) for every root tree in the batch.

        Returns:
            tree_fitness_list: list of fitness dicts (one per root)
            tree_regimes: dict counting regime occurrences
            dead_wrong_roots: list of (root_idx, root_node) for dead-wrong trees
            row_to_root: list mapping each train_batch row to its root index
        """
        # Map train_batch rows back to root indices using the level_node_list
        per_tree_stats = collect_node_logprobs_rewards_from_batch(
            train_batch, level_node_list, self.max_tree_depth
        )

        tree_fitness_list = []
        tree_regimes = {'dead_correct': 0, 'dead_wrong': 0, 'stale': 0, 'informative': 0}
        dead_wrong_roots = []

        for root_idx, root in enumerate(level_node_list[0]):
            leaf_rewards = collect_leaf_rewards(root)
            if root_idx in per_tree_stats:
                lps, rews = per_tree_stats[root_idx]
                fitness = compute_tree_fitness(leaf_rewards, lps, rews)
            else:
                fitness = compute_tree_fitness(leaf_rewards)
            regime = classify_tree(fitness)
            fitness['regime'] = regime
            tree_fitness_list.append(fitness)
            tree_regimes[regime] += 1
            if regime == 'dead_wrong':
                dead_wrong_roots.append((root_idx, root))

        # Build row_to_root mapping: walk the train batch in the same order
        # as _format_train_batch to track which row belongs to which root
        row_to_root = self._build_row_to_root_map(level_node_list)

        return tree_fitness_list, tree_regimes, dead_wrong_roots, row_to_root

    def _build_row_to_root_map(self, level_node_list):
        """Map each row in the formatted train batch to its root tree index.

        Mirrors the iteration order of _format_train_batch exactly.
        """
        # Build node -> root_idx mapping via DFS from each root
        node_to_root = {}
        for root_idx, root in enumerate(level_node_list[0]):
            def _tag(node, ri=root_idx):
                node_to_root[id(node)] = ri
                for c in node.children:
                    _tag(c, ri)
            _tag(root)

        row_to_root = []
        for target_depth in range(1, self.max_tree_depth + 1):
            if target_depth >= len(level_node_list) or len(level_node_list[target_depth]) == 0:
                break
            for node in level_node_list[target_depth]:
                if not hasattr(node, 'bro_rewards'):
                    continue
                if max(node.bro_rewards) - min(node.bro_rewards) < 0.1:
                    continue
                row_to_root.append(node_to_root.get(id(node), -1))
        return row_to_root

    def _apply_fitness_weights(self, train_batch, tree_fitness_list, row_to_root):
        """Scale advantages by normalized per-tree fitness weights."""
        n_rows = train_batch.batch['advantages'].shape[0]
        fitness_scores = []
        for row_idx in range(n_rows):
            if row_idx < len(row_to_root):
                root_i = row_to_root[row_idx]
                if 0 <= root_i < len(tree_fitness_list):
                    fitness_scores.append(tree_fitness_list[root_i]['F'])
                else:
                    fitness_scores.append(1.0)
            else:
                fitness_scores.append(1.0)

        F_tensor = torch.tensor(fitness_scores, dtype=torch.float32)
        # Normalize so mean weight = 1 (preserves gradient scale)
        F_sum = F_tensor.sum()
        if F_sum > 1e-8:
            F_normalized = F_tensor / F_sum * len(F_tensor)
        else:
            F_normalized = torch.ones_like(F_tensor)

        # Scale advantages per row
        train_batch.batch['advantages'] = (
            train_batch.batch['advantages'] * F_normalized.unsqueeze(-1)
        )
        train_batch.batch['returns'] = (
            train_batch.batch['returns'] * F_normalized.unsqueeze(-1)
        )
        return train_batch

    def _find_shallowest_failure(self, root, epsilon=0.05):
        """BFS to find the shallowest node where ALL children have E_reward < epsilon."""
        from collections import deque
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if len(node.children) == 0:
                continue
            all_children_fail = all(c.E_reward < epsilon for c in node.children)
            if all_children_fail and node.depth < self.max_tree_depth:
                return node
            for child in node.children:
                queue.append(child)
        return None

    def _heal_dead_wrong_trees(self, dead_wrong_roots, level_node_list, k=4):
        """Critique-guided healing for dead-wrong trees.

        For each dead-wrong tree:
        1. Find shallowest failing depth
        2. Generate natural-language critique
        3. Generate k refined continuations conditioned on critique
        4. Graft as new children, score, and add to level_node_list

        Returns:
            healed_count: number of trees successfully healed
        """
        healed_count = 0

        for root_idx, root in dead_wrong_roots:
            # Step 1: Find shallowest failure point
            fail_node = self._find_shallowest_failure(root)
            if fail_node is None:
                continue

            # Step 2: Decode prefix text
            prefix_ids = fail_node.batch_dict['input_ids'][0]
            mask = fail_node.batch_dict['attention_mask'][0]
            # Only decode the non-padded tokens
            valid_ids = prefix_ids[mask.bool()]
            prefix_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)

            # Extract the original problem from the root
            root_ids = root.batch_dict['input_ids'][0]
            root_mask = root.batch_dict['attention_mask'][0]
            root_valid = root_ids[root_mask.bool()]
            problem_text = self.tokenizer.decode(root_valid, skip_special_tokens=True)

            # Step 3: Generate critique
            critique_prompt = (
                "You are a mathematical reasoning critic. "
                "A student attempted the following problem but got it wrong.\n\n"
                f"Problem: {problem_text}\n\n"
                f"Student's partial solution:\n{prefix_text}\n\n"
                "Identify the specific mathematical or logical error. "
                "Be precise about which step is wrong and why."
            )
            critique_text = self._generate_text(critique_prompt, temperature=0.3, max_tokens=256)
            if critique_text is None:
                continue

            # Step 4: Generate k refined continuations
            refine_prompt = (
                f"Problem: {problem_text}\n\n"
                f"A student's incorrect attempt:\n{prefix_text}\n\n"
                f"Critique of the error:\n{critique_text}\n\n"
                "Provide a corrected solution continuing from where the error was found. "
                "Show your work step by step."
            )

            # Generate k continuations from the failing node
            refinements_grafted = 0
            for _ in range(k):
                refined_text = self._generate_text(refine_prompt, temperature=0.6, max_tokens=self.config.actor_rollout_ref.rollout.step_length)
                if refined_text is None:
                    continue

                # Tokenize the refined continuation
                refined_ids = self.tokenizer.encode(refined_text, add_special_tokens=False, return_tensors='pt')
                step_len = self.config.actor_rollout_ref.rollout.step_length

                # Truncate or pad to step_length
                if refined_ids.shape[1] > step_len:
                    refined_ids = refined_ids[:, :step_len]

                # Build the full input_ids: prefix + refined continuation
                prefix = fail_node.batch_dict['input_ids']  # (1, prefix_len)
                full_ids = torch.cat([prefix, refined_ids.to(prefix.device)], dim=1)

                # Pad/truncate to expected length
                expected_len = self.config.data.max_prompt_length + self.config.actor_rollout_ref.rollout.step_length * (fail_node.depth + 1)
                if full_ids.shape[1] < expected_len:
                    pad_len = expected_len - full_ids.shape[1]
                    full_ids = torch.cat([full_ids, torch.full((1, pad_len), self.tokenizer.pad_token_id, device=full_ids.device)], dim=1)
                elif full_ids.shape[1] > expected_len:
                    full_ids = full_ids[:, :expected_len]

                full_mask = (full_ids != self.tokenizer.pad_token_id).long()
                full_pos = torch.clamp(torch.cumsum(full_mask, dim=1) - 1, min=0)

                start_idx = self.config.data.max_prompt_length + self.config.actor_rollout_ref.rollout.step_length * fail_node.depth
                end_idx = start_idx + step_len
                responses = full_ids[:, start_idx:end_idx]

                # Score the refined leaf with reward function
                # Build a minimal batch for reward scoring
                score_batch = DataProto.from_single_dict({
                    'input_ids': full_ids,
                    'attention_mask': full_mask,
                    'position_ids': full_pos,
                    'prompts': full_ids[:, :start_idx],
                    'responses': responses,
                    'data_source': fail_node.batch_dict['data_source'],
                    'ability': fail_node.batch_dict['ability'],
                    'reward_model': fail_node.batch_dict['reward_model'],
                    'extra_info': fail_node.batch_dict['extra_info'],
                    'index': fail_node.batch_dict['index'],
                })
                try:
                    reward = self.reward_fn(score_batch).sum(-1)
                    reward_val = reward[0].item()
                except Exception as e:
                    print(f"[CATPO] Reward scoring failed for healed node: {e}")
                    continue

                # Check if the response ends with EOS
                eos_id = self.tokenizer.eos_token_id
                valid_resp_len = full_mask[0, -step_len:].sum().item()

                # Create child TreeNode
                child = tree.TreeNode(
                    batch_dict={
                        'input_ids': full_ids,
                        'attention_mask': full_mask,
                        'position_ids': full_pos,
                        'prompts': full_ids[:, :start_idx],
                        'responses': responses,
                        'data_source': fail_node.batch_dict['data_source'],
                        'ability': fail_node.batch_dict['ability'],
                        'reward_model': fail_node.batch_dict['reward_model'],
                        'extra_info': fail_node.batch_dict['extra_info'],
                        'index': fail_node.batch_dict['index'],
                    },
                    depth=fail_node.depth + 1,
                    prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    step_length=self.config.actor_rollout_ref.rollout.step_length,
                    reward=reward_val,
                    eos_token_id=eos_id,
                )
                fail_node.add_child(child)
                fail_node.child_rewards.append(reward_val)

                # Add to level_node_list
                target_depth = fail_node.depth + 1
                if target_depth < len(level_node_list):
                    level_node_list[target_depth].append(child)

                refinements_grafted += 1

            if refinements_grafted > 0:
                healed_count += 1
                print(f"[CATPO] Healed tree {root_idx}: grafted {refinements_grafted} refined branches")

        return healed_count

    def _generate_text(self, prompt_text, temperature=0.6, max_tokens=256):
        """Generate text using the actor model via tokenize -> generate -> decode.

        Returns decoded text string, or None on failure.
        """
        try:
            input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt', truncation=True,
                                               max_length=self.config.data.max_prompt_length)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

            # Pad to max_prompt_length
            prompt_len = self.config.data.max_prompt_length
            if input_ids.shape[1] < prompt_len:
                pad_len = prompt_len - input_ids.shape[1]
                input_ids = torch.cat([
                    torch.full((1, pad_len), self.tokenizer.pad_token_id), input_ids
                ], dim=1)
                attention_mask = torch.cat([
                    torch.zeros(1, pad_len, dtype=torch.long), attention_mask
                ], dim=1)
                position_ids = torch.clamp(torch.cumsum(attention_mask, dim=1) - 1, min=0)
            elif input_ids.shape[1] > prompt_len:
                input_ids = input_ids[:, -prompt_len:]
                attention_mask = attention_mask[:, -prompt_len:]
                position_ids = torch.arange(prompt_len).unsqueeze(0)

            gen_batch = DataProto.from_single_dict({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            })
            gen_batch.meta_info['max_gen_tokens'] = max_tokens
            gen_batch.meta_info['sample_n'] = 1
            if temperature != 0.6:
                gen_batch.meta_info['temperature'] = temperature

            # Pad for DP
            gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
            response = self.actor_rollout_wg.generate_sequences(gen_batch)

            # Decode first (non-padded) response
            resp_ids = response.batch['responses'][0]
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"[CATPO] Text generation failed: {e}")
            return None

    def _compute_batch_tree_advantage(self, input_batch):
        batch_size = input_batch.batch['input_ids'].shape[0]

        # init tree
        level_node_list = [[] for _ in range(self.max_tree_depth+1)]
        for idx in range(batch_size):
            level_node_list[0].append(
                tree.TreeNode(
                    batch_dict={
                        'input_ids': input_batch.batch['input_ids'][idx:idx+1, :],
                        'attention_mask': input_batch.batch['attention_mask'][idx:idx+1, :],
                        'position_ids': input_batch.batch['position_ids'][idx:idx+1, :],
                        'data_source': input_batch.non_tensor_batch['data_source'][idx:idx+1],
                        'ability': input_batch.non_tensor_batch['ability'][idx:idx+1],
                        'reward_model': input_batch.non_tensor_batch['reward_model'][idx:idx+1],
                        'extra_info': input_batch.non_tensor_batch['extra_info'][idx:idx+1],
                        'index': input_batch.non_tensor_batch['index'][idx:idx+1],
                    },
                    depth=0,
                    prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    step_length=self.config.actor_rollout_ref.rollout.step_length,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            )
        
        level_node_list.append(level_node_list[0])

        for depth in range(1, self.max_tree_depth+1):
            print("## father depth: {} -------------".format(depth-1))
            # gen next depth
            depth_batch, pad_size = self._format_depth_input_batch(level_node_list[depth-1])

            if depth_batch is None:
                break

            # depth_batch.meta_info['max_gen_tokens'] = self.config.actor_rollout_ref.rollout.step_length
            depth_batch.meta_info['max_gen_tokens'] = self.config.data.max_response_length - self.config.actor_rollout_ref.rollout.step_length * (depth-1)

            if depth == 1:
                depth_batch.meta_info['sample_n'] = self.config.actor_rollout_ref.rollout.n
            else:
                depth_batch.meta_info['sample_n'] = self.config.actor_rollout_ref.rollout.n - 1
            

            depth_response_batch = self.actor_rollout_wg.generate_sequences(depth_batch)

            depth_batch_reward = self.reward_fn(depth_response_batch)
            depth_batch_reward = depth_batch_reward.sum(-1)


            # print("** input/output size:", depth, depth_batch.batch['input_ids'].shape, depth_response_batch.batch['responses'].shape)
            # print(depth_batch.meta_info['sample_n'], depth_response_batch.meta_info['sample_n'])
            # print("###### depth_batch--------------------------------")
            # print(depth_batch)
            # print("###### response_batch--------------------------------")
            # print(depth_response_batch)
            # exit()
            # print("** child num:", [len(node.children) for node in level_node_list[depth-1]])

            self._format_child_node(level_node_list, depth-1, depth_response_batch, depth_batch_reward, pad_size)
            # print("** child num:", [len(node.children) for node in level_node_list[depth-1]])

            # record next level
            for node in level_node_list[depth-1]:
                for child in node.children:
                    level_node_list[depth].append(child)
            print("** depth-{}-size: {}".format(depth, len(level_node_list[depth])))
            print("** pad_size:", pad_size)
            

        self._traversal_reward(level_node_list[0])
        self._traversal_adv(level_node_list[0])

        # format train batch
        train_batch = self._format_train_batch(level_node_list)
        if train_batch is None:
            print("## no valid train batch")
            return None, None, 0, {}

        # compute log prob
        old_log_prob = self.actor_rollout_wg.compute_log_prob(train_batch)
        train_batch = train_batch.union(old_log_prob)
        if self.use_reference_policy:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_batch)
            train_batch = train_batch.union(ref_log_prob)

        print("## batch size --------------------")
        print(train_batch.batch['input_ids'].shape[0])

        # ==============================================================
        # CATPO: Fitness diagnosis, healing, and weighting
        # ==============================================================
        fitness_metrics = {}
        enable_catpo = getattr(self.config.trainer, 'enable_catpo', False)

        if enable_catpo:
            # Phase 2: Compute per-tree fitness
            tree_fitness_list, tree_regimes, dead_wrong_roots, row_to_root = \
                self._compute_tree_fitness_all(level_node_list, train_batch)

            n_roots = len(level_node_list[0])
            fitness_metrics['fitness/mean_F'] = np.mean([f['F'] for f in tree_fitness_list])
            fitness_metrics['fitness/mean_H'] = np.mean([f['H'] for f in tree_fitness_list])
            fitness_metrics['fitness/mean_rho'] = np.mean([f['rho'] for f in tree_fitness_list])
            fitness_metrics['fitness/dead_correct_pct'] = tree_regimes['dead_correct'] / max(n_roots, 1)
            fitness_metrics['fitness/dead_wrong_pct'] = tree_regimes['dead_wrong'] / max(n_roots, 1)
            fitness_metrics['fitness/stale_pct'] = tree_regimes['stale'] / max(n_roots, 1)
            fitness_metrics['fitness/informative_pct'] = tree_regimes['informative'] / max(n_roots, 1)

            print(f"[CATPO] Fitness: F={fitness_metrics['fitness/mean_F']:.3f}, "
                  f"H={fitness_metrics['fitness/mean_H']:.3f}, "
                  f"rho={fitness_metrics['fitness/mean_rho']:.3f}")
            print(f"[CATPO] Regimes: {tree_regimes}")

            # Phase 3: Critique-guided healing of dead-wrong trees
            healed_count = 0
            enable_healing = getattr(self.config.trainer, 'enable_catpo_healing', False)
            if enable_healing and len(dead_wrong_roots) > 0:
                k = getattr(self.config.trainer, 'catpo_num_refinements', 4)
                healed_count = self._heal_dead_wrong_trees(dead_wrong_roots, level_node_list, k=k)

                if healed_count > 0:
                    # Re-propagate rewards and recompute advantages for healed trees
                    self._traversal_reward(level_node_list[0])
                    self._traversal_adv(level_node_list[0])

                    # Re-format train batch with healed nodes included
                    train_batch = self._format_train_batch(level_node_list)
                    if train_batch is None:
                        print("## no valid train batch after healing")
                        return None, None, 0, {}

                    # Recompute log probs for the new train batch
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(train_batch)
                    train_batch = train_batch.union(old_log_prob)
                    if self.use_reference_policy:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_batch)
                        train_batch = train_batch.union(ref_log_prob)

                    # Recompute fitness after healing
                    tree_fitness_list, tree_regimes, _, row_to_root = \
                        self._compute_tree_fitness_all(level_node_list, train_batch)

                    # Update metrics with post-healing values
                    fitness_metrics['fitness/post_heal_mean_F'] = np.mean([f['F'] for f in tree_fitness_list])
                    fitness_metrics['fitness/post_heal_dead_wrong_pct'] = tree_regimes['dead_wrong'] / max(n_roots, 1)
                    print(f"[CATPO] Post-healing: F={fitness_metrics['fitness/post_heal_mean_F']:.3f}, "
                          f"dead_wrong={tree_regimes['dead_wrong']}")

            fitness_metrics['fitness/healed_count'] = healed_count

            # Phase 4: Apply fitness weights to advantages
            train_batch = self._apply_fitness_weights(train_batch, tree_fitness_list, row_to_root)
            print(f"[CATPO] Applied fitness weights to {train_batch.batch['advantages'].shape[0]} rows")

        ## statistics
        # compute the mean acc of train batch
        train_batch_acc_list = [root_node.E_reward for root_node in level_node_list[0]]
        mean_train_batch_acc = np.mean(train_batch_acc_list).item()
        # compute the total response token nums
        response_mask = train_batch.batch['attention_mask'][:, -train_batch.batch['responses'].shape[-1]:]
        response_token_num = torch.sum(response_mask, dim=-1).tolist()
        batch_response_token_num = np.sum(response_token_num).item()

        return train_batch, mean_train_batch_acc, batch_response_token_num, fitness_metrics


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # print(f'Loaded checkpoint from {self.config.trainer.default_local_dir}')
        print(f'Current global step: {self.global_steps}')

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        self.total_train_response_token_num = 0

        # set max tree depth
        assert self.config.data.max_response_length % self.config.actor_rollout_ref.rollout.step_length == 0
        self.max_tree_depth = self.config.data.max_response_length // self.config.actor_rollout_ref.rollout.step_length
        print(f'Max tree depth: {self.max_tree_depth}')


        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                        dtype=object)

                
                with _timer('step', timing_raw):
                    batch, train_batch_acc, batch_response_token_num, fitness_metrics = \
                        self._compute_batch_tree_advantage(batch)

                    self.total_train_response_token_num += batch_response_token_num

                    metrics['batch/acc'] = train_batch_acc
                    metrics['batch/batch_response_token_num'] = batch_response_token_num
                    metrics['batch/total_train_response_token_num'] = self.total_train_response_token_num
                    metrics.update(fitness_metrics)
                    print("## train batch response token num: {}".format(batch_response_token_num))

                    if batch is None:
                        print("## no valid train batch")
                        continue

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    print("## batch_size before update actor: ", batch.batch['input_ids'].shape[0])
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    print("## successfully update actor...")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # exit()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
