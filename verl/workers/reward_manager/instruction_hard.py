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

from verl import DataProto
from verl.utils.reward_score.instruction import instruction_compute_score, instruction_val_compute_score
import torch
from collections import defaultdict


class Instruction_hard_RewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source'
                 ,max_resp_len=None,overlong_buffer_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

        if compute_score =='instruction':
            self.compute_score = instruction_compute_score
            self.mode = 'train'
        else:
            self.compute_score = instruction_val_compute_score
            self.mode = 'val'


    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        ac_count=0
        c_count=0
        score_l=0
        already_print_data_sources = {}
        already_print=0
        constraint_response_lst = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['ground_truth']

            constraints = ground_truth['instruction_id_list']
            constraint_response_lst.append({"cc": constraints,'rl': valid_response_length})
        
            # data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # extra_info = data_item.non_tensor_batch.get('extra_info', None)

            if self.mode == "train":
                score = self.compute_score(response_str, ground_truth)
            elif self.mode == "val":
                all_right,a_len_c,len_c,score = self.compute_score(response_str, ground_truth)
                ac_count+=a_len_c
                c_count+=len_c
                score_l +=all_right

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score


            reward_tensor[i, valid_response_length - 1] = reward


            # if data_source not in already_print_data_sources:
                # already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                # print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[val reward score]", score)

        if return_dict:
            if self.mode=='train':
                return  {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "constraint_response_lst":constraint_response_lst
            }
            elif self.mode=='val':
                score_c =ac_count/c_count
                score_l = score_l/len(data)
                return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "score_c": score_c,
                "score_l": score_l,
                "constraint_response_lst":constraint_response_lst
            }
        else:
            if self.mode=='train':
                return reward_tensor,constraint_response_lst
            elif self.mode=='val':
                score_c =ac_count/c_count
                score_l = score_l/len(data)
                return reward_tensor,score_c,score_l
            

